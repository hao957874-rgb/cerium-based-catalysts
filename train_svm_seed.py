import pandas as pd
import numpy as np
import os
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 1. 设置工作状态与读取数据
print("="*65)
file_path = r'D:\vsshujubao\CB\data\SVM\数据集2.xlsx'
print(f"正在读取数据文件: {file_path}")

df = pd.read_excel(file_path)

if '序号' in df.columns:
    df = df.drop(columns=['序号'])

target_col = 'CB转化率'
X_raw = df.drop(columns=[target_col])
y_raw = df[target_col].values
feature_names = X_raw.columns.tolist()

# 2. 多Seed稳定性验证参数设置
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,42, 20, 30, 50, 123, 256, 512, 1024, 2048, 4096]

seed_metrics = []

# 3. 定义网格搜索选参空间
param_grid = {
    'kernel': ['rbf'],
    'C': [10],
    'gamma': [0.12],
    'epsilon': [0.05]
}
grid = list(ParameterGrid(param_grid))

print(f"数据处理完毕。总数据量: {X_raw.shape[0]} 行")
print(f"准备进行多Seed稳定性验证，共 {len(seeds)} 个随机种子。")
print(f"搜索空间包含 {len(grid)} 种组合。")
print("-" * 65)

# 记录全局最好的模型用于后续的 SHAP 分析
best_global_r2 = -float('inf')
best_global_model = None
best_global_scaler = None
best_global_X_test = None
best_global_X_test_scaled = None
best_global_X_temp_scaled = None
best_global_seed = None

from sklearn.model_selection import KFold

for seed in tqdm(seeds, desc="多Seed运行进度"):
    # 第一步：先把 20% 的 Test 切分出来锁着，避免干扰
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=seed
    )
    
    best_val_rmse_seed = float('inf')
    best_params_seed = None
    
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    X_cv_arr = X_temp.values if hasattr(X_temp, 'values') else X_temp
    y_cv_arr = y_temp.values if hasattr(y_temp, 'values') else y_temp
    
    # 仍进行网格搜索以便未来可扩充超参数
    for params in grid:
        fold_val_rmse = []
        for train_index, val_index in kf.split(X_cv_arr):
            X_kf_tr, X_kf_val = X_cv_arr[train_index], X_cv_arr[val_index]
            y_kf_tr, y_kf_val = y_cv_arr[train_index], y_cv_arr[val_index]
            
            # 严防折间数据泄露
            kf_scaler = StandardScaler()
            X_kf_tr_scaled = kf_scaler.fit_transform(X_kf_tr)
            X_kf_val_scaled = kf_scaler.transform(X_kf_val)
            
            model = SVR(**params)
            model.fit(X_kf_tr_scaled, y_kf_tr)
            
            val_pred = model.predict(X_kf_val_scaled)
            fold_val_rmse.append(np.sqrt(mean_squared_error(y_kf_val, val_pred)))
            
        mean_val_rmse = np.mean(fold_val_rmse)
        if mean_val_rmse < best_val_rmse_seed:
            best_val_rmse_seed = mean_val_rmse
            best_params_seed = params
            
    # 测试集准备
    scaler_final = StandardScaler()
    X_temp_scaled = scaler_final.fit_transform(X_temp)
    X_test_scaled = scaler_final.transform(X_test)
    
    # 拟合该Seed下最好的模型
    final_model = SVR(**best_params_seed)
    final_model.fit(X_temp_scaled, y_temp)
    
    # 盲测评估
    test_pred = final_model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    
    seed_metrics.append({
        'seed': seed,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    })
    
    # 记录总体表现最佳的一组作为SHAP样本
    if test_r2 > best_global_r2:
        best_global_r2 = test_r2
        best_global_model = final_model
        best_global_scaler = scaler_final
        best_global_X_test = X_test
        best_global_X_test_scaled = X_test_scaled
        best_global_X_temp_scaled = X_temp_scaled
        best_global_seed = seed

# 汇总输出
print("\n" + "="*80)
print(f"--- {len(seeds)}个不同 Seed 的独立盲测结果汇总 ---")
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

# ==================== 多Seed稳定性验证可视化 ====================
# 设置中英文字体
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

seeds_str = [str(res['seed']) for res in seed_metrics]
r2_arr = np.array(r2_list)
rmse_arr = np.array(rmse_list)

font_en = {'family': 'Times New Roman'}
output_dir = os.path.dirname(file_path)

label_font = {'family': 'Times New Roman', 'size': 18, 'weight': 'bold'}
title_font_zh = {'family': ['Times New Roman', 'SimSun'], 'size': 20, 'weight': 'bold'}
tick_font = 'Times New Roman'

# 图1：所有 seed 的 Test R² 柱状图
fig1 = plt.figure(figsize=(8, 6), dpi=300)
ax1 = plt.gca()
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
bars1 = ax1.bar(seeds_str, r2_arr, color='#5B9BD5', edgecolor='black', width=0.6, zorder=3)
ax1.set_title('Test R² 随随机种子验证表现', fontdict=title_font_zh)
ax1.set_xlabel('Random Seed', fontdict=label_font)
ax1.set_ylabel('Test R²', fontdict=label_font)
ax1.tick_params(axis='x', rotation=45, labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
mean_r2 = np.mean(r2_arr)
std_r2 = np.std(r2_arr)
ax1.axhline(mean_r2, color='red', linestyle='--', linewidth=2, label=f'Mean R²: {mean_r2:.4f}', zorder=4)
max_r2 = max(float(np.max(r2_arr)), float(mean_r2))
for idx, bar in enumerate(bars1):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + max_r2 * 0.015, f'{yval:.3f}', ha='center', va='bottom', rotation=45, fontdict={'family': 'Times New Roman', 'size': 13, 'weight': 'bold'})
ax1.set_xlim(-1.0, len(seeds_str) - 0.2)
ax1.set_ylim(0, max_r2 * 1.22)
ax1.legend(prop={'family': 'Times New Roman', 'size': 18})
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontname('Times New Roman')
for label in ax1.get_xticklabels():
    label.set_horizontalalignment('right')
    label.set_rotation_mode('anchor')
ax1.set_ylim(top=ax1.get_ylim()[1] * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "svm_multi_seed_r2_bar.png"), dpi=300, bbox_inches='tight')
plt.close(fig1)

# 图2：所有 seed 的 Test RMSE 柱状图
fig2 = plt.figure(figsize=(8, 6), dpi=300)
ax2 = plt.gca()
ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
bars2 = ax2.bar(seeds_str, rmse_arr, color='#F4D03F', edgecolor='black', width=0.6, zorder=3)
ax2.set_title('Test RMSE 随随机种子验证表现', fontdict=title_font_zh)
ax2.set_xlabel('Random Seed', fontdict=label_font)
ax2.set_ylabel('Test RMSE', fontdict=label_font)
ax2.tick_params(axis='x', rotation=45, labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
mean_rmse = np.mean(rmse_arr)
std_rmse = np.std(rmse_arr)
ax2.axhline(mean_rmse, color='red', linestyle='--', linewidth=2, label=f'Mean RMSE: {mean_rmse:.4f}', zorder=4)
max_rmse = max(float(np.max(rmse_arr)), float(mean_rmse))
for idx, bar in enumerate(bars2):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + max_rmse * 0.015, f'{yval:.3f}', ha='center', va='bottom', rotation=45, fontdict={'family': 'Times New Roman', 'size': 13, 'weight': 'bold'})
ax2.set_xlim(-1.0, len(seeds_str) - 0.2)
ax2.set_ylim(0, max_rmse * 1.22)
ax2.legend(prop={'family': 'Times New Roman', 'size': 18})
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontname('Times New Roman')
for label in ax2.get_xticklabels():
    label.set_horizontalalignment('right')
    label.set_rotation_mode('anchor')
ax2.set_ylim(top=ax2.get_ylim()[1] * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "svm_multi_seed_rmse_bar.png"), dpi=300, bbox_inches='tight')
plt.close(fig2)

# 图3：Test R² 的分布箱线图
fig3 = plt.figure(figsize=(8, 6), dpi=300)
ax3 = plt.gca()
ax3.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
sns.boxplot(y=r2_arr, ax=ax3, color='#D3D3D3', width=0.3, zorder=2)
sns.stripplot(y=r2_arr, ax=ax3, color='#E77C6E', size=6, jitter=True, alpha=0.9, linewidth=1, edgecolor='black', zorder=3)
ax3.set_title('Distribution of Test R²', fontdict=title_font_zh)
ax3.set_ylabel('Test R²', fontdict=label_font)
ax3.set_xlabel('SVR Model', fontdict=label_font)
ax3.set_xticks([0])
ax3.set_xticklabels([''])
ax3.tick_params(axis='y', labelsize=14)
for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontname('Times New Roman')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "svm_multi_seed_r2_box.png"), dpi=300, bbox_inches='tight')
plt.close(fig3)

# 图4：Test RMSE 与 Test R² 的散点图
fig4 = plt.figure(figsize=(8, 6), dpi=300)
ax4 = plt.gca()
ax4.grid(linestyle='--', alpha=0.5, zorder=0)
ax4.scatter(rmse_arr, r2_arr, color='#4EB9AA', edgecolor='black', s=50, zorder=3)
for i, txt in enumerate(seeds_str):
    ax4.annotate(txt, (rmse_arr[i], r2_arr[i]), xytext=(5, 5), textcoords='offset points', fontfamily='Times New Roman', fontsize=12)
ax4.set_title('Test RMSE vs Test R²', fontdict=title_font_zh)
ax4.set_xlabel('Test RMSE', fontdict=label_font)
ax4.set_ylabel('Test R²', fontdict=label_font)
ax4.tick_params(axis='both', labelsize=14)

# 拟合趋势线
z = np.polyfit(rmse_arr, r2_arr, 1)
p = np.poly1d(z)
x_trend = np.linspace(min(rmse_arr), max(rmse_arr), 100)
ax4.plot(x_trend, p(x_trend), "r--", label=f"Trend: y = {z[0]:.4f}x + {z[1]:.4f}")
ax4.legend(prop={'family': 'Times New Roman', 'size': 14})
for label in ax4.get_xticklabels() + ax4.get_yticklabels():
    label.set_fontname('Times New Roman')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "svm_multi_seed_scatter.png"), dpi=300, bbox_inches='tight')
plt.close(fig4)

print(f"多Seed稳定性图表(四张)已保存至: {output_dir}")
# ================================================================

print(f"\n提取表现最好的一组 (Seed={best_global_seed}, R2={best_global_r2:.4f}) 进入 SHAP 分析...")

# 8. SHAP 特征重要性分析
print("\n--- 正在进行 SHAP 特征重要性分析 (SVM 计算量较大，请耐心等待) ---")
background = shap.kmeans(best_global_X_temp_scaled, 100)
explainer = shap.KernelExplainer(best_global_model.predict, background)

test_samples = min(best_global_X_test_scaled.shape[0], 200)
idx_test = np.random.choice(best_global_X_test_scaled.shape[0], test_samples, replace=False)
X_test_explainer = best_global_X_test_scaled[idx_test]
X_vis = best_global_X_test.iloc[idx_test].copy()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    shap_values = explainer.shap_values(X_test_explainer, nsamples=100)

shap_values_arr = np.array(shap_values)

mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
sorted_indices = np.argsort(mean_abs_shap)[::-1]

print("\n--- 各特征的平均绝对 SHAP 贡献度 (重要性按降序排列) ---")
for idx in sorted_indices:
    print(f"{feature_names[idx]:<20} : {mean_abs_shap[idx]:.5f}")
print("="*80)
print("所有流程已执行完毕。")
