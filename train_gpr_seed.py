import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 1. 设置工作状态与读取数据
print("="*65)
file_path = r'D:\vsshujubao\CB\data\GPR\数据集2.xlsx'
print(f"正在读取数据文件: {file_path}")

df = pd.read_excel(file_path)

if '序号' in df.columns:
    df = df.drop(columns=['序号'])

target_col = 'CB转化率'
X_raw = df.drop(columns=[target_col])
y_raw = df[target_col].values
feature_names = X_raw.columns.tolist()

# 2. 多Seed稳定性验证参数设置
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42, 20, 30, 50, 123, 256, 512, 1024, 2048, 4096]
seed_metrics = []

# 3. 定义网格搜索参数
param_grid = {
    'kernel': [
    # C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)),
    C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=1.5),
    C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=2.5),  # 新增
    # C(1.0, (1e-3, 1e3)) * RationalQuadratic(1.0, alpha=0.1),
    # C(1.0, (1e-3, 1e3)) * RationalQuadratic(1.0, alpha=1.0),
    ],
    'alpha': [0.13],         # 噪声水平（类似于正则化项）
    'n_restarts_optimizer': [0]         # 优化器重启次数
    # 'kernel': [
    # # C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)),
    # C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=1.5),
    # C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=2.5),  # 新增
    # C(1.0, (1e-3, 1e3)) * RationalQuadratic(1.0, alpha=0.1),
    # C(1.0, (1e-3, 1e3)) * RationalQuadratic(1.0, alpha=1.0),
    # ],
    # 'alpha': [0.08, 0.1, 0.12, 0.15, 0.2],         # 噪声水平（类似于正则化项）
    # 'n_restarts_optimizer': [0]         # 优化器重启次数    
}
grid = list(ParameterGrid(param_grid))

print(f"数据处理完毕。总数据量: {X_raw.shape[0]} 行")
print(f"准备进行多Seed稳定性验证，共 {len(seeds)} 个随机种子。")
print(f"搜索空间包含 {len(grid)} 种组合。\n")
print("-" * 65)

# 记录全局最好的模型用于后续的 SHAP 分析
best_global_r2 = -float('inf')
best_global_model = None
best_global_scaler = None
best_global_X_test = None
best_global_X_test_scaled = None
best_global_X_train_scaled = None
best_global_seed = None

# 为了在显示结果时能清晰表明核函数类型
def get_kernel_name(kernel):
    k_str = str(kernel)
    if "RBF" in k_str: return "RBF"
    if "Matern" in k_str: return "Matern"
    if "RationalQuadratic" in k_str: return "RatQuad"
    return "Unknown"

for seed in tqdm(seeds, desc="多Seed运行进度"):
    # 第一步：切分数据集 80% Train / 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=seed
    )
    
    X_train_vals = X_train.values
    X_test_vals = X_test.values

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    best_val_rmse_seed = float('inf')
    best_params_seed = None
    
    # 5. 网格搜索验证
    for params in grid:
        f_val_rmse = []
        for train_idx, val_idx in kf.split(X_train_vals):
            X_kf_train_raw, X_kf_val_raw = X_train_vals[train_idx], X_train_vals[val_idx]
            y_kf_train, y_kf_val = y_train[train_idx], y_train[val_idx]
            
            # 折内独立归一化
            inner_scaler = StandardScaler()
            X_kf_train = inner_scaler.fit_transform(X_kf_train_raw)
            X_kf_val = inner_scaler.transform(X_kf_val_raw)
            
            model = GaussianProcessRegressor(
                kernel=clone(params['kernel']),
                alpha=params['alpha'],
                n_restarts_optimizer=params['n_restarts_optimizer'],
                random_state=seed,
                normalize_y=True
            )
            model.fit(X_kf_train, y_kf_train)
            val_pred = model.predict(X_kf_val)
            f_val_rmse.append(np.sqrt(mean_squared_error(y_kf_val, val_pred)))
            
        mean_v_rmse = np.mean(f_val_rmse)
        
        if mean_v_rmse < best_val_rmse_seed:
            best_val_rmse_seed = mean_v_rmse
            best_params_seed = params
            
    # 测试集准备
    final_scaler = StandardScaler()
    X_train_final_scaled = final_scaler.fit_transform(X_train_vals)
    X_test_final_scaled = final_scaler.transform(X_test_vals)

    # 用选出的最佳参数在当前种子整个80%数据拟合
    final_model = GaussianProcessRegressor(
        kernel=clone(best_params_seed['kernel']),
        alpha=best_params_seed['alpha'],
        n_restarts_optimizer=best_params_seed['n_restarts_optimizer'],
        random_state=seed,
        normalize_y=True
    )
    final_model.fit(X_train_final_scaled, y_train)

    # 盲测预测
    test_pred = final_model.predict(X_test_final_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    
    seed_metrics.append({
        'seed': seed,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    })
    
    # 记录总体表现最佳的一组作SHAP分析
    if test_r2 > best_global_r2:
        best_global_r2 = test_r2
        best_global_model = final_model
        best_global_scaler = final_scaler
        best_global_X_test = X_test
        best_global_X_test_scaled = X_test_final_scaled
        best_global_X_train_scaled = X_train_final_scaled
        best_global_seed = seed

# 汇总输出
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

# ================= 新增：多 seed 稳定性验证的可视化模块 =================
print("\n--- 正在生成多 Seed 稳定性验证可视化图表 (四张独立的图) ---")
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

output_dir = r'D:\vsshujubao\CB\data\GPR\SHAP_plots_GPR_seed'
os.makedirs(output_dir, exist_ok=True)

seed_strs = [str(res['seed']) for res in seed_metrics]
r2_arr = np.array(r2_list)
rmse_arr = np.array(rmse_list)
mean_r2, std_r2 = np.mean(r2_arr), np.std(r2_arr)
mean_rmse, std_rmse = np.mean(rmse_arr), np.std(rmse_arr)

title_font = {'family': 'Times New Roman', 'size': 20, 'weight': 'bold'}
label_font = {'family': 'Times New Roman', 'size': 18, 'weight': 'bold'}
tick_font = 'Times New Roman'

# 图1：所有 seed 的 Test R² 柱状图
fig1 = plt.figure(figsize=(8, 6), dpi=300)
ax1 = plt.gca()
bars1 = ax1.bar(seed_strs, r2_arr, color='#5D9BCA', edgecolor='black', width=0.6, zorder=3)
ax1.axhline(mean_r2, color='red', linestyle='--', linewidth=2, zorder=4, label=f'Mean R²: {mean_r2:.4f}')
ax1.set_xlim(-1.0, len(seed_strs)-0.2)
ax1.set_title('Test R² 随随机种子验证表现', fontsize=20, fontweight='bold', fontfamily='SimSun')
ax1.set_xlabel('Random Seed', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
ax1.set_ylabel('Test R²', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
ax1.tick_params(axis='x', rotation=45, labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
    tick.set_fontname(tick_font)
    tick.set_fontsize(14)
for tick in ax1.get_xticklabels():
    tick.set_horizontalalignment('right')
    tick.set_rotation_mode('anchor')
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax1.legend(loc='best', prop={'family': 'Times New Roman', 'size': 18})
for bar in bars1:
    yval = bar.get_height()
    offset = ax1.get_ylim()[1] * 0.015
    ax1.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.3f}', ha='center', va='bottom', fontfamily='Times New Roman', fontsize=13, fontweight='bold', rotation=45)
ax1.set_ylim(top=ax1.get_ylim()[1] * 1.22)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gpr_multi_seed_r2_bar.png"), dpi=300, bbox_inches='tight')
plt.close(fig1)

# 图2：所有 seed 的 Test RMSE 柱状图
fig2 = plt.figure(figsize=(8, 6), dpi=300)
ax2 = plt.gca()
bars2 = ax2.bar(seed_strs, rmse_arr, color='#F4D03F', edgecolor='black', width=0.6, zorder=3)
ax2.axhline(mean_rmse, color='red', linestyle='--', linewidth=2, zorder=4, label=f'Mean RMSE: {mean_rmse:.4f}')
ax2.set_xlim(-1.0, len(seed_strs)-0.2)
ax2.set_title('Test RMSE 随随机种子验证表现', fontsize=20, fontweight='bold', fontfamily='SimSun')
ax2.set_xlabel('Random Seed', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
ax2.set_ylabel('Test RMSE', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
ax2.tick_params(axis='x', rotation=45, labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
    tick.set_fontname(tick_font)
    tick.set_fontsize(14)
for tick in ax2.get_xticklabels():
    tick.set_horizontalalignment('right')
    tick.set_rotation_mode('anchor')
ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax2.legend(loc='best', prop={'family': 'Times New Roman', 'size': 18})
for bar in bars2:
    yval = bar.get_height()
    offset = ax2.get_ylim()[1] * 0.015
    ax2.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.3f}', ha='center', va='bottom', fontfamily='Times New Roman', fontsize=13, fontweight='bold', rotation=45)
ax2.set_ylim(top=ax2.get_ylim()[1] * 1.22)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gpr_multi_seed_rmse_bar.png"), dpi=300, bbox_inches='tight')
plt.close(fig2)

# 图3：Test R² 的分布箱线图
fig3 = plt.figure(figsize=(8, 6), dpi=300)
ax3 = plt.gca()
sns.boxplot(y=r2_arr, ax=ax3, color='#D3D3D3', width=0.3, zorder=2)
sns.stripplot(y=r2_arr, ax=ax3, color='#E77C6E', size=6, jitter=True, zorder=3)
ax3.set_title('Distribution of Test R²', fontdict=title_font)
ax3.set_xlabel('GPR Model', fontdict=label_font)
ax3.set_ylabel('Test R²', fontdict=label_font)
ax3.set_xticks([0])
ax3.set_xticklabels(['']) # 保留中心占位刻度避免突兀
for tick in ax3.get_xticklabels() + ax3.get_yticklabels():
    tick.set_fontname(tick_font)
    tick.set_fontsize(14)
    tick.set_fontsize(14)
ax3.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gpr_multi_seed_r2_box.png"), dpi=300, bbox_inches='tight')
plt.close(fig3)

# 图4：Test RMSE 与 Test R² 的散点图
fig4 = plt.figure(figsize=(8, 6), dpi=300)
ax4 = plt.gca()
ax4.scatter(rmse_arr, r2_arr, color='#4EB9AA', s=60, zorder=3)
# 线性趋势线
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
ax4.tick_params(axis='both', labelsize=14)
for tick in ax4.get_xticklabels() + ax4.get_yticklabels():
    tick.set_fontname(tick_font)
    tick.set_fontsize(14)
ax4.grid(True, linestyle='--', alpha=0.7, zorder=0)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gpr_multi_seed_scatter.png"), dpi=300, bbox_inches='tight')
plt.close(fig4)
print(f"多 Seed 稳定性验证四张独立图表已保存至:\n  -> {output_dir}")
# ========================================================================

print(f"\n提取表现最好的一组 (Seed={best_global_seed}, R2={best_global_r2:.4f}) 进入 SHAP 分析...")

# 9. SHAP 特征重要性分析
print("\n--- 正在进行 SHAP 特征重要性分析 (模型偏大，请耐心等待) ---")
background = shap.kmeans(best_global_X_train_scaled, 50)
explainer = shap.KernelExplainer(best_global_model.predict, background)

X_shap_eval_scaled = best_global_X_test_scaled
# 修复1: 统一 X_vis 的格式与特征标度，避免 DataFrame 原始数值画图与 Scaled 版 SHAP 出现对齐或语义报错问题
X_vis = pd.DataFrame(best_global_X_test_scaled, columns=feature_names)

# 将 tqdm 加入 SHAP 评估阶段
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    shap_values = explainer.shap_values(X_shap_eval_scaled, nsamples='auto')

# 修复3: GPR 单输出回归的 shap_values 已经是 2D ndarray，无需冗余转换
shap_values_arr = shap_values

# 配置文件输出路径
output_dir = r'D:\vsshujubao\CB\data\GPR\SHAP_plots_GPR_seed'
os.makedirs(output_dir, exist_ok=True)

# 解决画图中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# (1) 绘制 Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_arr, X_vis, show=False)
summary_path = os.path.join(output_dir, "gpr_shap_summary_plot.png")
plt.tight_layout()
plt.savefig(summary_path, dpi=300)
plt.close()
print(f"\nSHAP 摘要蜂群图 (Summary Plot) 已保存至:\n  -> {summary_path}")

# (2) 绘制 最重要特征的 Dependence Plot
mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
sorted_indices = np.argsort(mean_abs_shap)[::-1]
# 修复2: 强制转换为标准 Python int 避免部分版本中 numpy.int64 索引报错
top_feature = feature_names[int(sorted_indices[0])]

plt.figure(figsize=(8, 6))
shap.dependence_plot(int(sorted_indices[0]), shap_values_arr, X_vis, show=False)
dep_path = os.path.join(output_dir, f"gpr_shap_dependence_plot_{top_feature.replace('/', '_').replace(':', '')}.png")
plt.tight_layout()
plt.savefig(dep_path, dpi=300)
plt.close()
print(f"SHAP 依赖图 (Dependence Plot) [基于特征:{top_feature}] 已保存至:\n  -> {dep_path}")

# (3) 打印各个特征的贡献度排序
print("\n--- 各特征的平均绝对 SHAP 贡献度 (重要性按降序排列) ---")
for idx in sorted_indices:
    print(f"{feature_names[idx]:<20} : {mean_abs_shap[idx]:.5f}")
print("="*80)
print("GPR 算法所有流程已执行完毕。")
