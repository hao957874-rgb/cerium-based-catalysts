import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 解决全局所有图表的字体：数字和字母用新罗马(Times New Roman)，汉字用宋体(SimSun)
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
# 解决宋体会导致负号显示为空白/乱码的问题
plt.rcParams['axes.unicode_minus'] = False

# 1. 设置工作状态与读取数据
print("="*65)
file_path = r'D:\vsshujubao\CB\data\SVM\新数据集2.xlsx'
print(f"正在读取数据文件: {file_path}")

df = pd.read_excel(file_path)

if '序号' in df.columns:
    df = df.drop(columns=['序号'])

# 将列名中的 cm3 替换为包含上标的 cm³
df.rename(columns=lambda x: str(x).replace('cm3', 'cm³'), inplace=True)

target_col = 'CB转化率'
X_raw = df.drop(columns=[target_col])
y_raw = df[target_col].values
feature_names = X_raw.columns.tolist()

# 2. 划分盲测集和内部交叉验证集：Train+Val(拟合与选参) / Test(盲测)
# 第一步：先把 20% 的 Test 切分出来锁着，直到最终评估前绝对不看
from sklearn.model_selection import KFold
X_temp, X_test, y_temp, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# 3. 为最终合并重训步骤准备一套提取自 80% 全局的完整归一器
scaler_final = StandardScaler()
X_temp_scaled = scaler_final.fit_transform(X_temp)
X_test_scaled = scaler_final.transform(X_test)

# 4. 定义网格搜索选参空间
param_grid = {
    # 'kernel': ['rbf'],
    # 'C': [1, 5, 10, 20, 50, 100],
    # 'gamma': ['scale', 'auto', 0.1, 0.05, 0.01, 0.005,0.12],
    # 'epsilon': [0.01,0.03, 0.05, 0.1, 0.2]
    'kernel': ['rbf'],
    'C': [10],
    'gamma': [0.12],
    'epsilon': [0.05]
}
grid = list(ParameterGrid(param_grid))

print(f"数据处理完毕。\n内部 CV 交叉验证集 (80%): {X_temp.shape[0]} 行\n外置终极盲测集 (20%): {X_test.shape[0]} 行")
print("-" * 65)
print("开始在 80% 内部数据上执行 5-Fold 交叉验证选参 (全程不触碰 Test)...")
print(f"搜索空间包含 {len(grid)} 种组合。\n")

results = []
best_val_rmse = float('inf')
best_params = None
best_fold_details = None

# 使用 KFold 将内部 80% 数据分成 5 份进行交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_cv_arr = X_temp.values if hasattr(X_temp, 'values') else X_temp
y_cv_arr = y_temp.values if hasattr(y_temp, 'values') else y_temp

# 使用 tqdm 追踪网格搜索进度
for params in tqdm(grid, desc="5-Fold交叉验证选参"):
    fold_tr_rmse, fold_val_rmse = [], []
    fold_tr_r2, fold_val_r2 = [], []
    
    for train_index, val_index in kf.split(X_cv_arr):
        X_kf_tr, X_kf_val = X_cv_arr[train_index], X_cv_arr[val_index]
        y_kf_tr, y_kf_val = y_cv_arr[train_index], y_cv_arr[val_index]
        
        # 严防折间数据泄露，每个折内部重新拟合出完全独立的 MinMaxScaler等
        kf_scaler = StandardScaler()
        X_kf_tr_scaled = kf_scaler.fit_transform(X_kf_tr)
        X_kf_val_scaled = kf_scaler.transform(X_kf_val)
        
        model = SVR(**params)
        model.fit(X_kf_tr_scaled, y_kf_tr)
        
        val_pred = model.predict(X_kf_val_scaled)
        tr_pred  = model.predict(X_kf_tr_scaled)
        
        fold_val_rmse.append(np.sqrt(mean_squared_error(y_kf_val, val_pred)))
        fold_tr_rmse.append(np.sqrt(mean_squared_error(y_kf_tr, tr_pred)))
        fold_val_r2.append(r2_score(y_kf_val, val_pred))
        fold_tr_r2.append(r2_score(y_kf_tr, tr_pred))
        
    mean_val_rmse = np.mean(fold_val_rmse)
    
    results.append({
        'params': params,
        'train_rmse': np.mean(fold_tr_rmse),
        'val_rmse': mean_val_rmse,
        'train_r2': np.mean(fold_tr_r2),
        'val_r2': np.mean(fold_val_r2)
    })
    
    # 根据 5 折平均 Val RMSE 敲定最稳参数
    if mean_val_rmse < best_val_rmse:
        best_val_rmse = mean_val_rmse
        best_params   = params
        best_fold_details = {
            'val_rmse': fold_val_rmse,
            'val_r2': fold_val_r2
        }

print("\n" + "="*80)
print(f"基于 5-Fold 交叉验证选出的最稳健参数: {best_params}")
print("="*80)

print("\n--- 不同参数组合在 5 折上的平均表现 (Top 10) ---")
print(f"{'Params (C, gamma, eps)':<35} | {'CV Tr RMSE':<12} | {'CV Val RMSE':<12} | {'CV Tr R2':<8} | {'CV Val R2':<8}")
print("-" * 88)
results_sorted = sorted(results, key=lambda x: x['val_rmse'])
for res in results_sorted[:10]:
    p_str = f"C={res['params']['C']}, g={res['params']['gamma']}, e={res['params']['epsilon']}"
    best_flag = "(*Opt)" if res['params'] == best_params else ""
    print(f"{p_str:<35} | {res['train_rmse']:<12.4f} | {res['val_rmse']:<12.4f} | {res['train_r2']:<8.4f} | {res['val_r2']:<8.4f} {best_flag}")

# 5. 合并先验知识并终极评估测试盲测集
print("\n" + "="*80)
print("正在将选定的最优参数拟合到合并的完整历史数据 (Train+Val) 上...")
best_model = SVR(**best_params)
# 将 X_temp, y_temp (即最初的 80% 集合) 用于最终拟合以利用完整知识
best_model.fit(X_temp_scaled, y_temp)

test_pred = best_model.predict(X_test_scaled)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

print("--- 测试集 (20%盲测用未见数据) 最终客观性能评估 ---")
print(f"Test RMSE (Loss) = {test_rmse:.4f}")
print(f"Test R2          = {test_r2:.4f}")
print("="*80)

# ======= 插入新增功能: 模型性能可视化 =======
print("\n--- 正在生成模型性能可视化图表 ---")
plot_dir = r'D:\vsshujubao\CB\data\SVM\Model_plots2'
os.makedirs(plot_dir, exist_ok=True)

# 1. 提取绘图数据
best_res = next(r for r in results if r['params'] == best_params)
cv_val_rmse = best_res['val_rmse']
cv_val_r2 = best_res['val_r2']

train_pred_full = best_model.predict(X_temp_scaled)
train_rmse_full = np.sqrt(mean_squared_error(y_temp, train_pred_full))
train_r2_full = r2_score(y_temp, train_pred_full)

# 2. 绘制柱状图 (Loss和R2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# 细化展示：全量训练、CV的每一折、CV平均、测试集
labels = ['Train\n(Full)', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'CV Avg', 'Test']
rmse_vals = [train_rmse_full] + best_fold_details['val_rmse'] + [cv_val_rmse, test_rmse]
r2_vals = [train_r2_full] + best_fold_details['val_r2'] + [cv_val_r2, test_r2]

# 为不同类别配置颜色（全量：蓝色，各折：浅金，平均：深金，测试：橙色）
colors = ['#5D98C1'] + ['#F4D03F']*5 + ['#D4AF37', '#E9826F']
bar_width = 0.5  # 降低柱子宽度使图片更美观

ax1.bar(labels, rmse_vals, color=colors, edgecolor='black', alpha=0.8, width=bar_width)
ax1.set_title('Loss (RMSE) Comparison', fontsize=20, fontweight='bold')
ax1.set_ylabel('RMSE Score', fontsize=18, fontweight='bold')
ax1.set_xlabel('Dataset Split', fontsize=18, fontweight='bold')
ax1.tick_params(axis='x', rotation=0, labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
rmse_max = max(rmse_vals)
rmse_offset = rmse_max * 0.015
for i, v in enumerate(rmse_vals):
    ax1.text(i, v + rmse_offset, f"{v:.3f}", ha='center', fontweight='bold', fontsize=13)
ax1.set_ylim(0, rmse_max * 1.22)

ax2.bar(labels, r2_vals, color=colors, edgecolor='black', alpha=0.8, width=bar_width)
ax2.set_title('R² Comparison', fontsize=20, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=18, fontweight='bold')
ax2.set_xlabel('Dataset Split', fontsize=18, fontweight='bold')
ax2.tick_params(axis='x', rotation=0, labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
r2_max = max(r2_vals)
r2_offset = r2_max * 0.015
r2_ymin = min(0, min(r2_vals) - 0.05)
ax2.set_ylim(r2_ymin, r2_max * 1.22)
for i, v in enumerate(r2_vals):
    ax2.text(i, v + r2_offset, f"{v:.3f}", ha='center', fontweight='bold', fontsize=13)

bars_path = os.path.join(plot_dir, "svm_metrics_bars.png")
plt.tight_layout()
plt.savefig(bars_path, dpi=300)
plt.close()
print(f"柱状图 (Loss & R2) 已保存至: {bars_path}")

# 3. 绘制真实值与预测值的散点图 (参考提供的图片风格)
plt.figure(figsize=(7, 7))

# 绘制训练集点
plt.scatter(y_temp, train_pred_full, color='#5D98C1', edgecolor='black', alpha=0.8, s=80, label='Train data')
# 绘制测试集点
plt.scatter(y_test, test_pred, color='#E9826F', edgecolor='black', alpha=0.8, s=80, label='Test data')

# 计算边界以绘制完美预测虚线
min_val = min(np.min(y_temp), np.min(y_test), np.min(train_pred_full), np.min(test_pred))
max_val = max(np.max(y_temp), np.max(y_test), np.max(train_pred_full), np.max(test_pred))
padding = (max_val - min_val) * 0.05
min_val -= padding
max_val += padding

# 画参考对角线
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=2, label='y = x (Ideal)')

# 图表装饰
plt.title('SVR', fontsize=20, fontweight='bold')
plt.xlabel('True Values', fontsize=18, fontweight='bold', color='#444444')
plt.ylabel('Predicted Values', fontsize=18, fontweight='bold', color='#444444')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.grid(True, linestyle='--', color='gray', alpha=0.3)
plt.legend(loc='upper left', fontsize=14, framealpha=0.9, edgecolor='lightgray', markerscale=1.1, prop={'size': 22})

# 右下角独立文本标注 R2（尽量模仿原图蓝/橙双色呈现）
text_x = max_val - padding
text_y_test = min_val + padding * 3
text_y_train = min_val + padding * 1.5
plt.text(text_x, text_y_test, f"Test R²: {test_r2:.4f}", fontsize=22, ha='right', va='bottom', color='#E9826F')
plt.text(text_x, text_y_train, f"Train R²: {train_r2_full:.4f}", fontsize=22, ha='right', va='bottom', color='#5D98C1')

scatter_path = os.path.join(plot_dir, "svm_true_vs_pred_scatter.png")
plt.tight_layout()
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"真实值-预测值散点图 已保存至: {scatter_path}")

# 8. SHAP 特征重要性分析
print("\n--- 正在进行 SHAP 特征重要性分析 (SVM 计算量较大，请耐心等待) ---")
# 对于 SVM 为避免计算时长爆炸，使用 KMeans 初始化背景数据，选取 50 个背景点代表全局
background = shap.kmeans(X_temp_scaled, 100)
explainer = shap.KernelExplainer(best_model.predict, background)

# 在最终真实的独立盲测集(Test Set)上提取样本以计算 SHAP 值，保证解释的客观严谨性
test_samples = min(X_test_scaled.shape[0], 200)
idx_test = np.random.choice(X_test_scaled.shape[0], test_samples, replace=False)
X_test_explainer = X_test_scaled[idx_test]
# X_vis 直接使用 X_test 对应的切片，提供真实物理特征尺度画图，修复原先基于 X_raw 误采错位索引的问题
X_vis = X_test.iloc[idx_test].copy()

# 将 tqdm 加入 SHAP 评估（KernelExplainer默认有内建tqdm，但为了控制体验限制 nsamples）
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    shap_values = explainer.shap_values(X_test_explainer, nsamples=100)

shap_values_arr = np.array(shap_values)

# 配置文件输出路径
output_dir = r'D:\vsshujubao\CB\data\SVM\SHAP_plots_SVM2'
os.makedirs(output_dir, exist_ok=True)

# (1) 绘制 Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_arr, X_vis, show=False)
ax_summary = plt.gca()
ax_summary.set_title('SHAP Summary Plot', fontsize=20, fontweight='bold', pad=12)
ax_summary.set_xlabel(ax_summary.get_xlabel(), fontsize=18, fontweight='bold')
ax_summary.set_ylabel(ax_summary.get_ylabel(), fontsize=18, fontweight='bold')
ax_summary.tick_params(axis='both', labelsize=16)
summary_path = os.path.join(output_dir, "svm_shap_summary_plot.png")
plt.tight_layout()
plt.savefig(summary_path, dpi=300)
plt.close()
print(f"\nSHAP 摘要蜂群图 (Summary Plot) 已保存至:\n  -> {summary_path}")

# (2) 绘制 SHAP 特征贡献度柱状图 (横向条形图)
plt.figure(figsize=(10, 8))
# 添加 plot_type="bar"，修改配色为您希望的暗红色 (Dark Red)
shap.summary_plot(shap_values_arr, X_vis, plot_type="bar", show=False, color="#8B0000")

ax = plt.gca()
# 删除原本图下方自带的 (average impact on model output magnitude) 标签
ax.set_xlabel("")
# 增加图名在上方
ax.set_title("SHAP Feature Contributions", fontsize=20, fontweight='bold', pad=15)
ax.set_xlabel('mean(|SHAP value|)', fontsize=18, fontweight='bold')
ax.set_ylabel('Features', fontsize=18, fontweight='bold')
ax.tick_params(axis='both', labelsize=16)

max_width = 0
# 给柱状图依次标上具体的数值
for p in ax.patches:
    width = p.get_width()
    if width > 0: # 避免标注空柱或负边
        # 将文本放置在柱形右侧的一点位置，使其不会重叠
        ax.text(width * 1.02, p.get_y() + p.get_height() / 2, 
            f'{width:.3f}', ha='left', va='center', fontsize=15, fontweight='bold')
        if width > max_width:
            max_width = width

# 拓展 x 轴的显示范围，留出大约 12% 的空间给右侧的数字，防止越过边框
if max_width > 0:
    ax.set_xlim(0, max_width * 1.12)

# 给图中内容在四周添加上黑色的全框闭合
for side in ['top', 'right', 'left', 'bottom']:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color('black')
    ax.spines[side].set_linewidth(1.2)

bar_path = os.path.join(output_dir, "svm_shap_bar_plot.png")
plt.tight_layout()
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"SHAP 贡献度柱状图 (Bar Plot) 已保存至:\n  -> {bar_path}")

# (3) 绘制 最重要特征的 Dependence Plot
mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
sorted_indices = np.argsort(mean_abs_shap)[::-1]
top_feature = feature_names[sorted_indices[0]]

plt.figure(figsize=(8, 6))
shap.dependence_plot(sorted_indices[0], shap_values_arr, X_vis, show=False)
dep_path = os.path.join(output_dir, f"svm_shap_dependence_plot_{top_feature}.png")
plt.tight_layout()
plt.savefig(dep_path, dpi=300)
plt.close()
print(f"SHAP 依赖图 (Dependence Plot) [基于特征:{top_feature}] 已保存至:\n  -> {dep_path}")

# 打出各个特征的贡献度排序
print("\n--- 各特征的平均绝对 SHAP 贡献度 (重要性按降序排列) ---")
for idx in sorted_indices:
    print(f"{feature_names[idx]:<20} : {mean_abs_shap[idx]:.5f}")
print("="*80)
print("所有流程已执行完毕。")
