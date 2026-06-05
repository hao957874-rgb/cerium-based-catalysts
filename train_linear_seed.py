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
import seaborn as sns
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

best_model = LinearModel(input_dim).to(device)
optimizer = optim.Adam(best_model.parameters(), lr=best_lr, weight_decay=best_wd)
criterion = nn.MSELoss()

best_model.train()
for epoch in range(best_epochs):
    optimizer.zero_grad()
    outputs = best_model(X_train_full_t)
    loss = criterion(outputs, y_train_full_t)
    loss.backward()
    optimizer.step()

# --- 6. 独立测试集性能评估 ---
best_model.eval()
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    test_pred_t = best_model(X_test_t)
    test_pred = test_pred_t.cpu().numpy()

test_rmse = math.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

print("\n" + "="*80)
print("--- 测试集 (20%未见数据) 最终性能评估 ---")
print(f"Test RMSE = {test_rmse:.4f}")
print(f"Test R2   = {test_r2:.4f}")
print("="*80)

# ================= 新增：多 seed 稳定性验证模块 =================
print("\n" + "="*80)
print("--- 准备进行多 Seed 稳定性验证 ---")
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42, 20, 30, 50, 123, 256, 512, 1024, 2048, 4096]
seed_metrics = []

best_global_r2 = -float('inf')
best_global_model = None
best_global_scaler = None
best_global_X_test_scaled = None
best_global_X_train_scaled = None
best_global_seed = None

for seed in tqdm(seeds, desc="多Seed运行进度"):
    # 1. 固定种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # 2. 切分数据集
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=seed
    )
    y_train_s = y_train_s.reshape(-1, 1)
    y_test_s = y_test_s.reshape(-1, 1)
    
    # 独立 fit_transform StandardScaler 防泄露
    scaler_s = StandardScaler()
    X_train_scaled_s = scaler_s.fit_transform(X_train_s)
    X_test_scaled_s = scaler_s.transform(X_test_s)
    
    kf_s = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_val_rmse_s = float('inf')
    best_params_s = None
    
    # 3. 5折交叉验证选参
    for params in grid:
        f_val_rmse_s = []
        for train_idx, val_idx in kf_s.split(X_train_scaled_s):
            X_kf_train = torch.tensor(X_train_scaled_s[train_idx], dtype=torch.float32).to(device)
            y_kf_train = torch.tensor(y_train_s[train_idx], dtype=torch.float32).to(device)
            X_kf_val = torch.tensor(X_train_scaled_s[val_idx], dtype=torch.float32).to(device)
            y_kf_val = torch.tensor(y_train_s[val_idx], dtype=torch.float32).to(device)
            
            model_s = LinearModel(input_dim).to(device)
            criterion_s = nn.MSELoss()
            optimizer_s = optim.Adam(model_s.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            
            model_s.train()
            for epoch in range(params['epochs']):
                optimizer_s.zero_grad()
                outputs = model_s(X_kf_train)
                loss = criterion_s(outputs, y_kf_train)
                loss.backward()
                optimizer_s.step()
                
            model_s.eval()
            with torch.no_grad():
                val_pred = model_s(X_kf_val).cpu().numpy()
                
            y_val_true = y_kf_val.cpu().numpy()
            f_val_rmse_s.append(math.sqrt(mean_squared_error(y_val_true, val_pred)))
            
        mean_v_rmse_s = np.mean(f_val_rmse_s)
        if mean_v_rmse_s < best_val_rmse_s:
            best_val_rmse_s = mean_v_rmse_s
            best_params_s = params
            
    # 4. 初始化最佳参数并做全量重训
    best_epochs_s = best_params_s['epochs']
    best_lr_s = best_params_s['lr']
    best_wd_s = best_params_s['weight_decay']
    
    X_train_full_t_s = torch.tensor(X_train_scaled_s, dtype=torch.float32).to(device)
    y_train_full_t_s = torch.tensor(y_train_s, dtype=torch.float32).to(device)
    
    final_model_s = LinearModel(input_dim).to(device)
    optimizer_full_s = optim.Adam(final_model_s.parameters(), lr=best_lr_s, weight_decay=best_wd_s)
    criterion_full_s = nn.MSELoss()
    
    final_model_s.train()
    for epoch in range(best_epochs_s):
        optimizer_full_s.zero_grad()
        outputs = final_model_s(X_train_full_t_s)
        loss = criterion_full_s(outputs, y_train_full_t_s)
        loss.backward()
        optimizer_full_s.step()
        
    # 5. 在独立测试集上评估
    final_model_s.eval()
    X_test_t_s = torch.tensor(X_test_scaled_s, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_pred_s = final_model_s(X_test_t_s).cpu().numpy()
        
    test_rmse_s = math.sqrt(mean_squared_error(y_test_s, test_pred_s))
    test_r2_s = r2_score(y_test_s, test_pred_s)
    
    # 6. 将结果记录
    seed_metrics.append({
        'seed': seed,
        'test_rmse': test_rmse_s,
        'test_r2': test_r2_s
    })
    
    # 7. 更新全局表现最优模型
    if test_r2_s > best_global_r2:
        best_global_r2 = test_r2_s
        best_global_model = final_model_s
        best_global_scaler = scaler_s
        best_global_X_test_scaled = X_test_scaled_s
        best_global_X_train_scaled = X_train_scaled_s
        best_global_seed = seed

# 打印汇总表
print("\n" + "="*80)
print("--- 20个不同 Seed 的独立盲测结果汇总 ---")
print(f"{'Seed':<10} | {'Test RMSE':<15} | {'Test R2':<15}")
print("-" * 45)

r2_list = []
rmse_list = []
for res in seed_metrics:
    print(f"{res['seed']:<10} | {res['test_rmse']:<15.4f} | {res['test_r2']:<15.4f}")
    r2_list.append(res['test_r2'])
    rmse_list.append(res['test_rmse'])

print("-" * 45)
print(f"Mean (均值) | {np.mean(rmse_list):<15.4f} | {np.mean(r2_list):<15.4f}")
print(f"Std  (方差) | {np.std(rmse_list):<15.4f} | {np.std(r2_list):<15.4f}")
print("="*80)

# ================= 可视化模块 =================
print("\n--- 正在生成多 Seed 稳定性验证可视化图表 ---")
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

multi_seed_output_dir = r'D:\vsshujubao\CB\data\liner\SHAP_plots_Linear_seed'
os.makedirs(multi_seed_output_dir, exist_ok=True)

seed_strs = [str(res['seed']) for res in seed_metrics]
r2_arr = np.array(r2_list)
rmse_arr = np.array(rmse_list)
mean_r2, std_r2 = np.mean(r2_arr), np.std(r2_arr)
mean_rmse, std_rmse = np.mean(rmse_arr), np.std(rmse_arr)

title_font = {'family': 'Times New Roman', 'size': 20, 'weight': 'bold'}
label_font = {'family': 'Times New Roman', 'size': 18, 'weight': 'bold'}
tick_font = 'Times New Roman'
title_font_zh = {'family': ['Times New Roman', 'SimSun'], 'size': 20, 'weight': 'bold'}

# 图1
plt.figure(figsize=(8, 6), dpi=300)
ax1 = plt.gca()
bars1 = ax1.bar(seed_strs, r2_arr, color='#5B9BD5', edgecolor='black', width=0.6, zorder=3)
ax1.axhline(mean_r2, color='red', linestyle='--', linewidth=2, zorder=4, label=f'Mean R²: {mean_r2:.4f}')
ax1.set_xlim(-1.0, len(seed_strs)-0.2)
ax1.legend(prop={'family': 'Times New Roman', 'size': 18})
ax1.set_title('Test R² 随随机种子验证表现', fontdict=title_font_zh)
ax1.set_xlabel('Random Seed', fontdict=label_font)
ax1.set_ylabel('Test R²', fontdict=label_font)
ax1.tick_params(axis='x', rotation=45, labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
    tick.set_fontname(tick_font)
for tick in ax1.get_xticklabels():
    tick.set_horizontalalignment('right')
    tick.set_rotation_mode('anchor')
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
for bar in bars1:
    yval = bar.get_height()
    y_offset = ax1.get_ylim()[1] * 0.015
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + y_offset, f'{yval:.3f}', ha='center', va='bottom', fontdict={'family': 'Times New Roman', 'size': 13, 'weight': 'bold'}, rotation=45)
ax1.set_ylim(top=ax1.get_ylim()[1] * 1.22)
plt.tight_layout()
plt.savefig(os.path.join(multi_seed_output_dir, "linear_multi_seed_r2_bar.png"), dpi=300, bbox_inches='tight')
plt.close()

# 图2
plt.figure(figsize=(8, 6), dpi=300)
ax2 = plt.gca()
bars2 = ax2.bar(seed_strs, rmse_arr, color='#F4D03F', edgecolor='black', width=0.6, zorder=3)
ax2.axhline(mean_rmse, color='red', linestyle='--', linewidth=2, zorder=4, label=f'Mean RMSE: {mean_rmse:.4f}')
ax2.set_xlim(-1.0, len(seed_strs)-0.2)
ax2.legend(prop={'family': 'Times New Roman', 'size': 18})
ax2.set_title('Test RMSE 随随机种子验证表现', fontdict=title_font_zh)
ax2.set_xlabel('Random Seed', fontdict=label_font)
ax2.set_ylabel('Test RMSE', fontdict=label_font)
ax2.tick_params(axis='x', rotation=45, labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
    tick.set_fontname(tick_font)
for tick in ax2.get_xticklabels():
    tick.set_horizontalalignment('right')
    tick.set_rotation_mode('anchor')
ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
for bar in bars2:
    yval = bar.get_height()
    y_offset = ax2.get_ylim()[1] * 0.015
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + y_offset, f'{yval:.3f}', ha='center', va='bottom', fontdict={'family': 'Times New Roman', 'size': 13, 'weight': 'bold'}, rotation=45)
ax2.set_ylim(top=ax2.get_ylim()[1] * 1.22)
plt.tight_layout()
plt.savefig(os.path.join(multi_seed_output_dir, "linear_multi_seed_rmse_bar.png"), dpi=300, bbox_inches='tight')
plt.close()

# 图3
plt.figure(figsize=(8, 6), dpi=300)
ax3 = plt.gca()
sns.boxplot(y=r2_arr, ax=ax3, color='#D3D3D3', width=0.3, zorder=2)
sns.stripplot(y=r2_arr, ax=ax3, color='#E77C6E', size=6, jitter=True, zorder=3)
ax3.set_title('Distribution of Test R²', fontdict=title_font)
ax3.set_xlabel('Linear Model', fontdict=label_font)
ax3.set_ylabel('Test R²', fontdict=label_font)
ax3.set_xticks([0])
ax3.set_xticklabels([''])
for tick in ax3.get_xticklabels() + ax3.get_yticklabels():
    tick.set_fontname(tick_font)
    tick.set_fontsize(14)
ax3.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(multi_seed_output_dir, "linear_multi_seed_r2_box.png"), dpi=300, bbox_inches='tight')
plt.close()

# 图4
plt.figure(figsize=(8, 6), dpi=300)
ax4 = plt.gca()
ax4.scatter(rmse_arr, r2_arr, color='#4EB9AA', s=60, zorder=3)
z = np.polyfit(rmse_arr, r2_arr, 1)
p = np.poly1d(z)
x_trend = np.linspace(min(rmse_arr), max(rmse_arr), 100)
ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}', zorder=2)
for i, txt in enumerate(seed_strs):
    ax4.annotate(txt, (rmse_arr[i], r2_arr[i]), textcoords="offset points", xytext=(0,5), ha='center', fontfamily='Times New Roman', fontsize=13, fontweight='bold')
ax4.set_title('Test RMSE vs Test R²', fontdict=title_font)
ax4.set_xlabel('Test RMSE', fontdict=label_font)
ax4.set_ylabel('Test R²', fontdict=label_font)
ax4.legend(prop={'family': 'Times New Roman', 'size': 18})
for tick in ax4.get_xticklabels() + ax4.get_yticklabels():
    tick.set_fontname(tick_font)
    tick.set_fontsize(14)
ax4.grid(True, linestyle='--', alpha=0.7, zorder=0)

plt.tight_layout()
plt.savefig(os.path.join(multi_seed_output_dir, "linear_multi_seed_scatter.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"四张独立的稳定性验证图表已保存至:\n  -> {multi_seed_output_dir}")
# ================================================================

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

# 解决画图中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# (1) 绘制 Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_arr, X_test_scaled, feature_names=feature_names, show=False)
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
