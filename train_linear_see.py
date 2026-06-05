import os
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import math
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- 1. 数据读取与预处理 ---
print("="*80)
# 若读取其他文件，请修改此处文件名
file_path = r'D:\vsshujubao\CB\data\liner\数据集2.xlsx'
print(f"正在读取数据文件: {file_path}")

df = pd.read_excel(file_path)

if '序号' in df.columns:
    df = df.drop(columns=['序号'])

# 将列名中的 cm3 替换为包含上标的 cm³ (同 SVM 一致，保持整齐)
df.rename(columns=lambda x: str(x).replace('cm3', 'cm³'), inplace=True)

target_col = 'CB转化率'
X_raw = df.drop(columns=[target_col])
y_raw = df[target_col].values
feature_names = np.array(X_raw.columns.tolist())

# 划分训练集和测试集 (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# 标准化特征 (线性模型对特征尺度极度敏感，必须对特征进行标准化)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# 检测并设置 GPU 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的训练设备: {device}")

# --- 2. 定义 PyTorch 线性模型 ---
class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        # 纯线性映射 (无需激活函数)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# --- 3. 定义网格搜索参数 ---
param_grid = {
    'lr': [0.04],          # 学习率
    'weight_decay': [0.0003],      # L2正则化(防过拟合)
    'epochs': [600]              # 训练轮数
}

# 生成所有参数组合
keys, values = zip(*param_grid.items())
grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"数据处理完毕。训练集规模：{X_train_scaled.shape} | 测试集规模：{X_test_scaled.shape}")
print("-" * 80)
print(f"开始执行 线性模型 网格搜索与 5 折交叉验证...")
print(f"搜索空间包含 {len(grid)} 种参数组合。\n")

input_dim = X_train_scaled.shape[1]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
best_val_rmse = float('inf')
best_params = None
best_fold_details = None

# --- 4. 交叉验证与网格搜索 ---
# 使用 tqdm 追踪网格搜索进度
for params in tqdm(grid, desc="网格搜索及5折交叉验证进度"):
    f_tr_rmse, f_val_rmse, f_tr_r2, f_val_r2 = [], [], [], []
    
    # 5折交叉验证
    for train_idx, val_idx in kf.split(X_train_scaled):
        # 转换为 Tensor 并移动到 GPU
        X_kf_train = torch.tensor(X_train_scaled[train_idx], dtype=torch.float32).to(device)
        y_kf_train = torch.tensor(y_train[train_idx], dtype=torch.float32).to(device)
        X_kf_val = torch.tensor(X_train_scaled[val_idx], dtype=torch.float32).to(device)
        y_kf_val = torch.tensor(y_train[val_idx], dtype=torch.float32).to(device)
        
        # 初始化模型、损失函数和优化器
        model = LinearModel(input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        
        # 训练过程
        model.train()
        for epoch in range(params['epochs']):
            optimizer.zero_grad()
            outputs = model(X_kf_train)
            loss = criterion(outputs, y_kf_train)
            loss.backward()
            optimizer.step()
            
        # 验证过程
        model.eval()
        with torch.no_grad():
            tr_pred = model(X_kf_train).cpu().numpy()
            val_pred = model(X_kf_val).cpu().numpy()
            
        y_tr_true = y_kf_train.cpu().numpy()
        y_val_true = y_kf_val.cpu().numpy()
        
        # 记录每折误差
        f_tr_rmse.append(math.sqrt(mean_squared_error(y_tr_true, tr_pred)))
        f_val_rmse.append(math.sqrt(mean_squared_error(y_val_true, val_pred)))
        f_tr_r2.append(r2_score(y_tr_true, tr_pred))
        f_val_r2.append(r2_score(y_val_true, val_pred))
        
    mean_v_rmse = np.mean(f_val_rmse)
    
    # 记录该参数组合的整体性能
    results.append({
        'params': params,
        'train_rmse': np.mean(f_tr_rmse),
        'val_rmse': mean_v_rmse,
        'train_r2': np.mean(f_tr_r2),
        'val_r2': np.mean(f_val_r2)
    })
    
    # 保存最佳参数
    if mean_v_rmse < best_val_rmse:
        best_val_rmse = mean_v_rmse
        best_params = params
        best_fold_details = {
            'tr_rmse': f_tr_rmse, 'val_rmse': f_val_rmse,
            'tr_r2': f_tr_r2, 'val_r2': f_val_r2
        }

# --- 5. 全量重训最佳模型 ---
# 提取最佳参数
best_epochs = best_params['epochs']
best_lr = best_params['lr']
best_wd = best_params['weight_decay']

print("\n" + "="*80)
print(f"网格搜索找到的最佳参数组合: {best_params}")
print("="*80)

# 集中输出最佳参数下 5折交叉验证 详细过程
print("\n--- 最佳参数下的 5 折交叉验证详细过程 ---")
print(f"{'Fold':<8} | {'Train RMSE':<12} | {'Val RMSE':<12} | {'Train R2':<10} | {'Val R2':<10}")
print("-" * 75)
for i in range(5):
    print(f"Fold {i+1:<3} | {best_fold_details['tr_rmse'][i]:<12.4f} | {best_fold_details['val_rmse'][i]:<12.4f} | {best_fold_details['tr_r2'][i]:<10.4f} | {best_fold_details['val_r2'][i]:<10.4f}")
print("-" * 75)
print(f"{'Mean':<8} | {np.mean(best_fold_details['tr_rmse']):<12.4f} | {np.mean(best_fold_details['val_rmse']):<12.4f} | {np.mean(best_fold_details['tr_r2']):<10.4f} | {np.mean(best_fold_details['val_r2']):<10.4f}")
print("="*75)

# 集中输出全网格搜索情况
print("\n--- 全网格搜索不同参数下的结果组合 (按 Val RMSE 升序排序) ---")
print(f"{'Params (lr, weight_decay, epochs)':<45} | {'Train RMSE':<12} | {'Val RMSE':<12} | {'Train R2':<10} | {'Val R2':<10}")
print("-" * 105)
results_sorted = sorted(results, key=lambda x: x['val_rmse'])
for res in results_sorted:
    p = res['params']
    p_str = f"lr={p['lr']}, wd={p['weight_decay']}, ep={p['epochs']}"
    best_flag = "(*Opt)" if p == best_params else ""
    print(f"{p_str:<45} | {res['train_rmse']:<12.4f} | {res['val_rmse']:<12.4f} | {res['train_r2']:<10.4f} | {res['val_r2']:<10.4f} {best_flag}")

# 在训练集的全部数据上进行最终模型拟合
X_train_full_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_full_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

best_model = LinearModel(input_dim).to(device)
optimizer = optim.Adam(best_model.parameters(), lr=best_lr, weight_decay=best_wd)
criterion = nn.MSELoss()

tr_losses, te_losses, tr_r2s, te_r2s = [], [], [], []

for epoch in range(best_epochs):
    best_model.train()
    optimizer.zero_grad()
    outputs = best_model(X_train_full_t)
    loss = criterion(outputs, y_train_full_t)
    loss.backward()
    optimizer.step()
    
    # 计算并记录当前 epoch 的 Loss 和 R2
    best_model.eval()
    with torch.no_grad():
        test_outputs = best_model(X_test_t)
        test_loss = criterion(test_outputs, y_test_t)
        
        # 记录 RMSE 而不是 MSE
        tr_losses.append(math.sqrt(loss.item()))
        te_losses.append(math.sqrt(test_loss.item()))
        tr_r2s.append(r2_score(y_train, outputs.cpu().numpy()))
        te_r2s.append(r2_score(y_test, test_outputs.cpu().numpy()))

# --- 6. 独立测试集性能评估 ---
best_model.eval()
with torch.no_grad():
    test_pred_t = best_model(X_test_t)
    train_pred_t = best_model(X_train_full_t)
    test_pred = test_pred_t.cpu().numpy()
    train_pred = train_pred_t.cpu().numpy()

test_rmse = math.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)
train_rmse_final = math.sqrt(mean_squared_error(y_train, train_pred))
train_r2_final = r2_score(y_train, train_pred)

print("\n" + "="*80)
print("--- 独立测试集 (20%未见数据) 最终性能评估 ---")
print(f"Test RMSE = {test_rmse:.4f}")
print(f"Test R2   = {test_r2:.4f}")
print("="*80)

# ------------------ 可视化部分 ------------------
plot_dir = r'D:\vsshujubao\CB\data\liner\Model_plots_Linear'
os.makedirs(plot_dir, exist_ok=True)

# 解决画图中文乱码及负号显示问题
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun', 'Songti SC', 'STSong']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 1. 学习曲线 (Training vs Testing Loss/R2)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

epochs_range = list(range(1, len(tr_losses)+1))
axes[0].plot(epochs_range, tr_losses, color='#2B4A9A', linewidth=2, label='Training Loss')
axes[0].plot(epochs_range, te_losses, color='#DE4242', linewidth=2, label='Test Loss')
axes[0].set_xlabel('Epochs', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Loss (RMSE)', fontsize=14, fontweight='bold')
axes[0].set_title('Training and Testing Loss Comparison', fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlim(left=0, right=max(epochs_range)+2)
axes[0].grid(True, linestyle='--', alpha=0.4)
axes[0].legend(fontsize=20, loc='upper right', prop={'family': 'Times New Roman', 'size': 20})
for spine in axes[0].spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

axes[1].plot(epochs_range, tr_r2s, color='#2B4A9A', linewidth=2, label='Training R²')
axes[1].plot(epochs_range, te_r2s, color='#DE4242', linewidth=2, label='Test R²')
axes[1].set_xlabel('Epochs', fontsize=14, fontweight='bold')
axes[1].set_ylabel('R² Score', fontsize=14, fontweight='bold')
axes[1].set_title('Training and Testing R² Comparison', fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlim(left=0, right=max(epochs_range)+2)
axes[1].grid(True, linestyle='--', alpha=0.4)
axes[1].legend(fontsize=20, loc='lower right', prop={'family': 'Times New Roman', 'size': 20})
for spine in axes[1].spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "linear_learning_curves.png"), dpi=300, bbox_inches='tight')
plt.close()

# 提取5折数据并绘制 8-Bar 性能柱状图
cv_rmses = best_fold_details['val_rmse']
cv_r2s = best_fold_details['val_r2']
avg_cv_rmse = np.mean(cv_rmses)
avg_cv_r2 = np.mean(cv_r2s)

labels = ['Train'] + [f'Fold {i+1}' for i in range(5)] + ['CV Avg', 'Test']
rmse_vals = [train_rmse_final] + list(cv_rmses) + [avg_cv_rmse, test_rmse]
r2_vals = [train_r2_final] + list(cv_r2s) + [avg_cv_r2, test_r2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
bar_width = 0.5
colors_rmse = ['#5D98C1'] + ['#F4D03F']*5 + ['#D4AF37', '#E9826F']
colors_r2 = ['#5D98C1'] + ['#F4D03F']*5 + ['#D4AF37', '#E9826F']

x = np.arange(len(labels))
b1 = ax1.bar(x, rmse_vals, color=colors_rmse, edgecolor='black', linewidth=1.2, width=bar_width)
ax1.set_title('Loss (RMSE) across Training, CV and Test', fontsize=20, fontweight='bold')
ax1.set_ylabel('RMSE', fontsize=18, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.set_ylim(0, max(rmse_vals) * 1.22)
for idx, rect in enumerate(b1):
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., h + max(rmse_vals)*0.015, f'{h:.4f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

b2 = ax2.bar(x, r2_vals, color=colors_r2, edgecolor='black', linewidth=1.2, width=bar_width)
ax2.set_title('R² Score across Training, CV and Test', fontsize=20, fontweight='bold')
ax2.set_ylabel('R²', fontsize=18, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=14)
ax2.tick_params(axis='y', labelsize=14)
max_r2_val = max(r2_vals) if max(r2_vals) > 0 else 0.1
ax2.set_ylim(min(0, min(r2_vals) * 1.22), max_r2_val * 1.22)
for idx, rect in enumerate(b2):
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h + max_r2_val*0.015, f'{h:.4f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
for spine in ax2.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "linear_performance_bars.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3. 散点图绘制
plt.figure(figsize=(10, 10))
plt.scatter(y_train, train_pred, color='#7fb2d5', edgecolors='black', s=100, alpha=0.9, linewidth=1.2, label='Train data')
plt.scatter(y_test, test_pred, color='#eb9690', edgecolors='black', s=100, alpha=0.9, linewidth=1.2, label='Test data')

min_val = min(y_train.min(), train_pred.min(), y_test.min(), test_pred.min())
max_val = max(y_train.max(), train_pred.max(), y_test.max(), test_pred.max())
margin = (max_val - min_val) * 0.05
limit_min = min_val - margin
limit_max = max_val + margin

plt.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', lw=2.5, label='y = x (Ideal)')
plt.xlim(limit_min, limit_max)
plt.ylim(limit_min, limit_max)
plt.title('Linear Regression', fontsize=20, fontweight='bold')
plt.xlabel('True Values', fontsize=18, fontweight='bold')
plt.ylabel('Predicted Values', fontsize=18, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left', markerscale=1.4, handletextpad=0.6, prop={'family': 'Times New Roman', 'size': 26})
plt.grid(True, linestyle='--', alpha=0.4)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.text(0.95, 0.05, f'Train R²: {train_r2_final:.4f}', transform=ax.transAxes, fontsize=30, color='#5D98C1', ha='right')
plt.text(0.95, 0.12, f'Test R²: {test_r2:.4f}', transform=ax.transAxes, fontsize=30, color='#E9826F', ha='right')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "linear_scatter_plot.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"评估图表已保存至: {plot_dir}")

# --- 7. SHAP 特征重要性分析 ---
print("\n--- 正在进行 SHAP 特征重要性分析 ---")
# SHAP 在计算 PyTorch 模型的依赖度时，可用 DeepExplainer
explainer = shap.DeepExplainer(best_model, X_train_full_t)
# 计算测试集的 SHAP 值
shap_values_t = explainer.shap_values(X_test_t)
# PyTorch 环境下 DeepExplainer 生成的是 Tensor/List，将其提取转为 NumPy array 且降维
if isinstance(shap_values_t, list):
    shap_values_arr = shap_values_t[0]
else:
    shap_values_arr = shap_values_t

# 输出目录设为代码所在的当前目录旁的 SHAP_plots 文件夹中可以按需修改
output_dir = r'D:\vsshujubao\CB\data\liner\SHAP_plots_Linear'
os.makedirs(output_dir, exist_ok=True)

# 解决画图中文乱码及负号显示问题
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun', 'Songti SC', 'STSong']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# (0) 绘制自定义 SHAP 柱状图 (深红色, 横向)
plt.figure(figsize=(10, 8))
if shap_values_arr.ndim == 3:
    shap_values_2d = shap_values_arr[:, :, 0]
else:
    shap_values_2d = shap_values_arr

mean_abs_impact = np.abs(shap_values_2d).mean(axis=0)
sorted_idx = np.argsort(mean_abs_impact)
pos_features = np.array(feature_names)[sorted_idx]
pos_impacts = mean_abs_impact[sorted_idx]

bars = plt.barh(np.arange(len(pos_features)), pos_impacts, color='#8B0000', edgecolor='black', linewidth=1)
# 强制让 Y 轴 (汉字特征名) 使用宋体
plt.yticks(np.arange(len(pos_features)), pos_features, fontname='SimSun', fontsize=16)
plt.xticks(fontname='Times New Roman', fontsize=16)
plt.xlabel('') # 隐藏 xlabel
plt.title('SHAP Feature Importance', fontname='Times New Roman', fontsize=20, fontweight='bold', pad=15)

max_width = max(pos_impacts)
plt.xlim(0, max_width * 1.25) # 防止文字超出边界
ax = plt.gca()
for i, rect in enumerate(bars):
    width = rect.get_width()
    ax.text(width * 1.02, rect.get_y() + rect.get_height() / 2,
            f'{width:.4f}', ha='left', va='center', 
            fontname='Times New Roman', fontsize=15, fontweight='bold')
            
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.tight_layout()
bar_path = os.path.join(output_dir, "linear_shap_bar_plot.png")
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSHAP 贡献率柱状图 已保存至: {bar_path}")

# (1) 绘制 Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_2d, X_test_scaled, feature_names=feature_names, max_display=len(feature_names), show=False)

fig = plt.gcf()
for ax_ in fig.axes:
    ax_.tick_params(axis='both', labelsize=16)
    ax_.set_title(ax_.get_title(), fontsize=20, fontweight='bold', pad=12)
    ax_.set_xlabel(ax_.get_xlabel(), fontsize=18, fontweight='bold')
    ax_.set_ylabel(ax_.get_ylabel(), fontsize=18, fontweight='bold')
    for label in ax_.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax_.get_yticklabels():
        text = label.get_text()
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            label.set_fontname('SimSun')
        else:
            label.set_fontname('Times New Roman')
    
    for axis_label in [ax_.xaxis.label, ax_.yaxis.label, ax_.title]:
        if axis_label:
            text = axis_label.get_text()
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                axis_label.set_fontname('SimSun')
            else:
                axis_label.set_fontname('Times New Roman')

summary_path = os.path.join(output_dir, "linear_shap_summary_plot.png")
plt.tight_layout()
plt.savefig(summary_path, dpi=300)
plt.close()
print(f"SHAP 摘要图 (Summary Plot) 已保存至: {summary_path}")

# (2) 绘制 Dependence Plot (提取重要性最高的特征)
# PyTorch 模型解释出的 shap_values 常常带有单一输出维度，形如 (N, F, 1)
# 须切片为纯 2D 数组 (N, F) 以保证后续的排序和 np.argsort 单行单列
if shap_values_arr.ndim == 3:
    shap_values_arr = shap_values_arr[:, :, 0]

mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
sorted_indices = np.argsort(mean_abs_shap)[::-1]
# 显式提取为纯正的标量数字（避免索引出 array 导致字符串变成 "['某特征']" 的形式）
top_feature_idx = int(np.ravel(sorted_indices)[0])
top_feature = str(feature_names[top_feature_idx])

import re
safe_feature_name = re.sub(r'[^\w\-]', '_', top_feature)

plt.figure(figsize=(8, 6))
shap.dependence_plot(top_feature, shap_values_arr, X_test_scaled, feature_names=feature_names, show=False)

fig = plt.gcf()
for ax_ in fig.axes:
    for label in ax_.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax_.get_yticklabels():
        label.set_fontname('Times New Roman')
        
    for axis_label in [ax_.xaxis.label, ax_.yaxis.label, ax_.title]:
        if axis_label:
            text = axis_label.get_text()
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                axis_label.set_fontname('SimSun')
            else:
                axis_label.set_fontname('Times New Roman')

dep_path = os.path.join(output_dir, f"linear_shap_dependence_plot_{safe_feature_name}.png")
plt.tight_layout()
plt.savefig(dep_path, dpi=300)
plt.close()
print(f"SHAP 依赖图 (Dependence Plot) [基于特征: {top_feature}] 已保存至: {dep_path}")

# (3) 打印排好序的每个特征贡献度
print("\n--- 各特征的平均绝对 SHAP 贡献度 (重要性按降序排列) ---")
for idx in sorted_indices:
    print(f"{feature_names[idx]:<20} : {mean_abs_shap[idx]:.5f}")

print("\n" + "="*80)
print("线性回归模型 (GPU) 训练与特征重要性分析全部完成！")
