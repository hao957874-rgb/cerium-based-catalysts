import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os
import re
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


# 1. 设置工作状态与读取数据
print("="*65)
file_path = r'D:\vsshujubao\CB\data\RF\数据集2.xlsx'
print(f"正在读取数据文件: {file_path}")

df = pd.read_excel(file_path)

if '序号' in df.columns:
    df = df.drop(columns=['序号'])

target_col = 'CB转化率'
X_raw = df.drop(columns=[target_col])
y_raw = df[target_col].values
# ④ 修复: 将 feature_names 转换为 numpy array 以支持全程安全向量化操作，避免列表底层索引隐患
feature_names = np.array(X_raw.columns.tolist())

# 2. 划分训练集和测试集 (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# 注意：随机森林对量纲和异常值不敏感，此处直接使用原始数据进行训练
# 这样不仅保留了真实物理/化学意义，画出来的 SHAP 依赖图横坐标也更直观。
X_train_arr = X_train.values
X_test_arr = X_test.values

# 3. 定义网格搜索参数
# 针对过拟合现象进行了调优：减小最大深度，增大分裂和叶子节点最小样本数，并引入特征采样参数 max_features
param_grid = {
    # 'n_estimators': [100, 200,300],
    # 'max_depth': [6,8,10],
    # 'min_samples_split': [5, 10,15],
    # 'min_samples_leaf': [4,6,8],
    # 'max_features': ['sqrt', 0.8,0.6]
    # "n_estimators": [400, 500, 600],
    # "max_depth": [10, 12, 14],
    # "min_samples_split": [2, 3, 5],
    # "min_samples_leaf": [1, 2, 3],
    # "max_features": [0.6, 0.7, 0.8]

    'max_samples': [0.9],      
    'n_estimators': [400],
    'max_depth': [12],
    'min_samples_split': [3],
    'min_samples_leaf': [2],
    'max_features': [0.7]     
}
grid = list(ParameterGrid(param_grid))

print(f"数据处理完毕。训练集规模：{X_train_arr.shape} | 测试集规模：{X_test_arr.shape}")
print("-" * 65)
print("开始执行 随机森林(RF) 网格搜索与 5 折交叉验证...")
print("提示：Scikit-Learn 的 RandomForest 原生主要走 CPU，已自动为您开启全核心并发 (n_jobs=-1) 以最大化速度。")
print(f"搜索空间包含 {len(grid)} 种组合。\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
best_val_rmse = float('inf')
best_params = None
best_fold_details = None
best_model = None

# 4. 网格搜索进度条
# 使用 tqdm 追踪网格搜索进度
for params in tqdm(grid, desc="RF网格搜索及5折交叉验证进度"):
    f_tr_rmse, f_val_rmse, f_tr_r2, f_val_r2 = [], [], [], []
    
    # 5折交叉验证
    for train_idx, val_idx in kf.split(X_train_arr):
        X_kf_train, X_kf_val = X_train_arr[train_idx], X_train_arr[val_idx]
        y_kf_train, y_kf_val = y_train[train_idx], y_train[val_idx]
        
        # 训练模型 (n_jobs=-1 使用全部CPU核心)
        model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
        model.fit(X_kf_train, y_kf_train)
        
        # 预测
        tr_pred = model.predict(X_kf_train)
        val_pred = model.predict(X_kf_val)
        
        # 记录每折误差
        f_tr_rmse.append(np.sqrt(mean_squared_error(y_kf_train, tr_pred)))
        f_val_rmse.append(np.sqrt(mean_squared_error(y_kf_val, val_pred)))
        f_tr_r2.append(r2_score(y_kf_train, tr_pred))
        f_val_r2.append(r2_score(y_kf_val, val_pred))
        
    mean_v_rmse = np.mean(f_val_rmse)
    
    # 记录该参数整体结果
    results.append({
        'params': params,
        'train_rmse': np.mean(f_tr_rmse),
        'val_rmse': mean_v_rmse,
        'train_r2': np.mean(f_tr_r2),
        'val_r2': np.mean(f_val_r2)
    })
    
    if mean_v_rmse < best_val_rmse:
        best_val_rmse = mean_v_rmse
        best_params = params
        best_fold_details = {
            'tr_rmse': f_tr_rmse, 'val_rmse': f_val_rmse,
            'tr_r2': f_tr_r2, 'val_r2': f_val_r2
        }

# 使用找到的最佳参数，在全部训练集上从头训练作为最终最佳模型
best_model = RandomForestRegressor(**best_params, n_jobs=-1, random_state=42)
best_model.fit(X_train_arr, y_train)

# 5. 打印5折交叉验证明细表 (集中输出)
print("\n" + "="*80)
print(f"网格搜索找到的最佳参数组合: {best_params}")
# ② 修复: 明确注明此时选出的这组分数的代表意义
print("注: 此处展示的 CV 指标是在 80% 的训练数据中执行划分预估得出的，真正的模型泛化性能请参考文末盲测集评估。")
print("="*80)

print("\n--- 最佳参数下的 5 折交叉验证详细过程 ---")
print(f"{'Fold':<8} | {'Train RMSE (Loss)':<17} | {'Val RMSE (Loss)':<17} | {'Train R2':<10} | {'Val R2':<10}")
print("-" * 80)
for i in range(5):
    print(f"Fold {i+1:<3} | {best_fold_details['tr_rmse'][i]:<17.4f} | {best_fold_details['val_rmse'][i]:<17.4f} | {best_fold_details['tr_r2'][i]:<10.4f} | {best_fold_details['val_r2'][i]:<10.4f}")
print("-" * 80)
print(f"{'Mean':<8} | {np.mean(best_fold_details['tr_rmse']):<17.4f} | {np.mean(best_fold_details['val_rmse']):<17.4f} | {np.mean(best_fold_details['tr_r2']):<10.4f} | {np.mean(best_fold_details['val_r2']):<10.4f}")
print("="*80)

# 6. 全网格搜索不同参数下的汇总表
print("\n--- 全网格搜索不同参数下的结果组合 (按 Val RMSE 升序排序) ---")
# ⑧ 修复: 加入 Overfit Gap (过拟合差值) 列帮助快速判断泛化落差
print(f"{'Params (n, max_d, split, leaf)':<45} | {'Train RMSE':<12} | {'Val RMSE':<12} | {'Overfit Gap':<12} | {'Train R2':<8} | {'Val R2':<8}")
print("-" * 110)
results_sorted = sorted(results, key=lambda x: x['val_rmse'])
for res in results_sorted:
    p = res['params']
    p_str = f"n={p['n_estimators']}, d={p['max_depth']}, s={p['min_samples_split']}, l={p['min_samples_leaf']}"
    best_flag = "(*Opt)" if res['params'] == best_params else ""
    overfit_gap = res['val_rmse'] - res['train_rmse']
    print(f"{p_str:<45} | {res['train_rmse']:<12.4f} | {res['val_rmse']:<12.4f} | {overfit_gap:<12.4f} | {res['train_r2']:<8.4f} | {res['val_r2']:<8.4f} {best_flag}")

# 7. 独立测试集表现
test_pred = best_model.predict(X_test_arr)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

print("\n" + "="*80)
print("--- 测试集 (20%未见数据) 最终性能评估 ---")
print(f"Test RMSE (Loss) = {test_rmse:.4f}")
print(f"Test R2          = {test_r2:.4f}")
print("="*80)

# 8. SHAP 特征重要性分析
print("\n--- 正在进行 SHAP 特征重要性分析 (使用 TreeExplainer，速度较快) ---")
# 随机森林可以直接使用专门对树模型优化的 TreeExplainer，不需要 Background 数据，计算极快
explainer = shap.TreeExplainer(best_model)
# ① 修复: 此处应当也转用纯 Numpy 数组，强制对齐保证不会静默抛引发异常格式或索引
shap_values_arr = explainer.shap_values(X_test_arr)

# ③ 修复: 彻底删除这块用于二分类任务截取维度、实则在回归单输出环境中会因为三维导致越界异常的危险逻辑。回归任务的shap_values_arr本身就是2D
# if len(shap_values_arr.shape) == 3:
#     shap_values_arr = shap_values_arr[:, :, 1]

# 生成输出路径
output_dir = r'D:\vsshujubao\CB\data\RF\SHAP_plots_RF'
os.makedirs(output_dir, exist_ok=True)

# 解决画图中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# (1) 绘制 Summary Plot (摘要图/蜂群图)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_arr, X_test_arr, feature_names=feature_names, show=False)
summary_path = os.path.join(output_dir, "rf_shap_summary_plot.png")
plt.tight_layout()
plt.savefig(summary_path, dpi=300)
plt.close()
print(f"\nSHAP 摘要蜂群图 (Summary Plot) 已保存至:\n  -> {summary_path}")

# (2) 绘制 最重要特征的 Dependence Plot (相关性图/依赖图)
mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
sorted_indices = np.argsort(mean_abs_shap)[::-1]
top_feature_idx = sorted_indices[0]
top_feature = feature_names[top_feature_idx]
safe_feature_name = re.sub(r'[^\w\-]', '_', top_feature)

plt.figure(figsize=(8, 6))
# ⑤ 修复: 依赖图特征传递特征名字符串，避免低版本 SHAP 显示索引的数值 BUG
shap.dependence_plot(top_feature, shap_values_arr, X_test_arr, feature_names=feature_names, show=False)
dep_path = os.path.join(output_dir, f"rf_shap_dependence_plot_{safe_feature_name}.png")
plt.tight_layout()
plt.savefig(dep_path, dpi=300)
plt.close()
print(f"SHAP 依赖图 (Dependence Plot) [基于特征:{top_feature}] 已保存至:\n  -> {dep_path}")

# (3) 打印所有特征的贡献度排序
print("\n--- 各特征的平均绝对 SHAP 贡献度 (重要性按降序排列) ---")
for idx in sorted_indices:
    print(f"{feature_names[idx]:<20} : {mean_abs_shap[idx]:.5f}")

print("="*80)
print("RF 模型所有流程已执行完毕。")

# ==============================================================================
# --- 新增模块：多 Seed 稳定性验证阶段 ---
# ==============================================================================
print("\n" + "="*80)
print(">>> 启动多 Seed 稳定性验证模块")
print("================================================================================")

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42, 20, 30, 50, 123, 256, 512, 1024, 2048, 4096]
multi_seed_results = []

def apply_dual_font_to_fig(fig):
    """中英文字体自适应函数，保证英数字为Times New Roman，汉字为SimSun(宋体)"""
    for ax in fig.axes:
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        texts_objs = [ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.get_legend():
            texts_objs.extend(ax.get_legend().get_texts())
        texts_objs.extend(ax.texts)
        for text_obj in labels + texts_objs:
            text = text_obj.get_text()
            if text:
                text_obj.set_fontfamily(['Times New Roman', 'SimSun'])

for seed in tqdm(seeds, desc="多 Seed 试验进度", colour="green"):
    # 1. 各 seed 控制数据集的划分 (不使用 scaler)
    X_tr_seed, X_te_seed, y_tr_seed, y_te_seed = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=seed
    )
    
    # 转换为 ndarray (原始值，无需归一化)
    X_tr_arr_seed = X_tr_seed.values if isinstance(X_tr_seed, pd.DataFrame) else X_tr_seed
    X_te_arr_seed = X_te_seed.values if isinstance(X_te_seed, pd.DataFrame) else X_te_seed
    y_tr_seed = y_tr_seed if isinstance(y_tr_seed, np.ndarray) else y_tr_seed.values
    
    best_val_rmse_seed = float('inf')
    best_params_seed = None
    
    # 用当前seed创建独立的KFold
    kf_seed = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    # 2. 网格搜索与交叉验证 (基于外层的 param_grid 与恒定的 n_jobs)
    for params in grid:
        fold_val_rmses = []
        for train_idx, val_idx in kf_seed.split(X_tr_arr_seed):
            X_kf_train, X_kf_val = X_tr_arr_seed[train_idx], X_tr_arr_seed[val_idx]
            y_kf_train, y_kf_val = y_tr_seed[train_idx], y_tr_seed[val_idx]
            
            # 使用当前 seed
            model = RandomForestRegressor(**params, n_jobs=-1, random_state=seed)
            model.fit(X_kf_train, y_kf_train)
            
            val_pred = model.predict(X_kf_val)
            fold_val_rmses.append(np.sqrt(mean_squared_error(y_kf_val, val_pred)))
        
        mean_val_rmse = np.mean(fold_val_rmses)
        if mean_val_rmse < best_val_rmse_seed:
            best_val_rmse_seed = mean_val_rmse
            best_params_seed = params
            
    # 3. 在独立测试集 (20%) 验证最终模型表现
    final_model_seed = RandomForestRegressor(**best_params_seed, n_jobs=-1, random_state=seed)
    final_model_seed.fit(X_tr_arr_seed, y_tr_seed)
    
    y_te_pred = final_model_seed.predict(X_te_arr_seed)
    seed_rmse = np.sqrt(mean_squared_error(y_te_seed, y_te_pred))
    seed_r2 = r2_score(y_te_seed, y_te_pred)
    
    multi_seed_results.append({
        'seed': seed,
        'rmse': seed_rmse,
        'r2': seed_r2
    })

# ------------------------------------------------------------------------------
# 结果汇总与打印表单
rmses_list = [res['rmse'] for res in multi_seed_results]
r2s_list = [res['r2'] for res in multi_seed_results]

rmse_mean, rmse_std = np.mean(rmses_list), np.std(rmses_list)
r2_mean, r2_std = np.mean(r2s_list), np.std(r2s_list)

print("\n" + "="*80)
print("--- 20个不同 Seed 的独立盲测结果汇总 ---")
print(f"{'Seed':<10} | {'Test RMSE':<15} | {'Test R2':<10}")
print("-" * 45)
for res in multi_seed_results:
    print(f"{res['seed']:<10} | {res['rmse']:<15.4f} | {res['r2']:<10.4f}")
print("-" * 45)
print(f"{'Mean (均值)':<10} | {rmse_mean:<15.4f} | {r2_mean:<10.4f}")
print(f"{'Std  (方差)':<10} | {rmse_std:<15.4f} | {r2_std:<10.4f}")
print("================================================================================\n")

# ------------------------------------------------------------------------------
# 绘制四大可视化图 (改为分离出四张独立的图)
multi_seed_dir = r'D:\vsshujubao\CB\data\RF\MultiSeed_plots_RF'
os.makedirs(multi_seed_dir, exist_ok=True)
seed_strs = [str(s) for s in seeds]

# 图1：Test R2 柱状图
fig1 = plt.figure(figsize=(8, 6), dpi=300)
ax1 = plt.gca()
bars1 = ax1.bar(seed_strs, r2s_list, color='#5D9BCA', edgecolor='black', width=0.6, zorder=3)
ax1.axhline(r2_mean, color='red', linestyle='--', linewidth=2, zorder=4, label=f'Mean R²: {r2_mean:.4f}')
ax1.set_xlim(-1.0, len(seed_strs)-0.2)
ax1.set_title('Test R² 随随机种子验证表现', fontsize=20, fontweight='bold')
ax1.set_xlabel('Random Seed', fontsize=18, fontweight='bold')
ax1.set_ylabel('Test R²', fontsize=18, fontweight='bold')
ax1.tick_params(axis='x', rotation=45, labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
for tick in ax1.get_xticklabels():
    tick.set_horizontalalignment('right')
    tick.set_rotation_mode('anchor')
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax1.legend(loc='best', prop={'family': 'Times New Roman', 'size': 18})
for bar in bars1:
    yval = bar.get_height()
    offset = ax1.get_ylim()[1] * 0.015
    ax1.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold', rotation=45)
ax1.set_ylim(top=ax1.get_ylim()[1] * 1.22)
apply_dual_font_to_fig(fig1)
plt.tight_layout()
plt.savefig(os.path.join(multi_seed_dir, "RF_MultiSeed_r2_bar.png"), dpi=300)
plt.close(fig1)

# 图2：Test RMSE 柱状图
fig2 = plt.figure(figsize=(8, 6), dpi=300)
ax2 = plt.gca()
bars2 = ax2.bar(seed_strs, rmses_list, color='#F4D03F', edgecolor='black', width=0.6, zorder=3)
ax2.axhline(rmse_mean, color='red', linestyle='--', linewidth=2, zorder=4, label=f'Mean RMSE: {rmse_mean:.4f}')
ax2.set_xlim(-1.0, len(seed_strs)-0.2)
ax2.set_title('Test RMSE 随随机种子验证表现', fontsize=20, fontweight='bold')
ax2.set_xlabel('Random Seed', fontsize=18, fontweight='bold')
ax2.set_ylabel('Test RMSE', fontsize=18, fontweight='bold')
ax2.tick_params(axis='x', rotation=45, labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
for tick in ax2.get_xticklabels():
    tick.set_horizontalalignment('right')
    tick.set_rotation_mode('anchor')
ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax2.legend(loc='best', prop={'family': 'Times New Roman', 'size': 18})
for bar in bars2:
    yval = bar.get_height()
    offset = ax2.get_ylim()[1] * 0.015
    ax2.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold', rotation=45)
ax2.set_ylim(top=ax2.get_ylim()[1] * 1.22)
apply_dual_font_to_fig(fig2)
plt.tight_layout()
plt.savefig(os.path.join(multi_seed_dir, "RF_MultiSeed_rmse_bar.png"), dpi=300)
plt.close(fig2)

# 图3：Test R2 箱线图 + 散点
fig3 = plt.figure(figsize=(8, 6), dpi=300)
ax3 = plt.gca()
sns.boxplot(y=r2s_list, ax=ax3, color='lightgray', showfliers=False, width=0.4, zorder=1)
sns.stripplot(y=r2s_list, ax=ax3, color='#E58066', size=8, jitter=True, alpha=0.8, edgecolor='black', linewidth=1, zorder=2)
ax3.set_title('Test R² 离散分布箱线图 (20 Seeds)', fontsize=20, fontweight='bold')
ax3.set_ylabel('Test R²', fontsize=18, fontweight='bold')
ax3.tick_params(axis='y', labelsize=14)
ax3.grid(axis='y', linestyle='--', alpha=0.5)
apply_dual_font_to_fig(fig3)
plt.tight_layout()
plt.savefig(os.path.join(multi_seed_dir, "RF_MultiSeed_r2_box.png"), dpi=300)
plt.close(fig3)

# 图4：Test RMSE vs Test R2 散点图 + 趋势线
fig4 = plt.figure(figsize=(8, 6), dpi=300)
ax4 = plt.gca()
ax4.scatter(rmses_list, r2s_list, color='#1ABC9C', s=80, edgecolor='black', alpha=0.8, zorder=3)
for res in multi_seed_results:
    ax4.text(res['rmse'], res['r2'] + 0.002, str(res['seed']), fontsize=13, fontweight='bold', ha='center', va='bottom')
z = np.polyfit(rmses_list, r2s_list, 1)
p = np.poly1d(z)
x_line = np.linspace(min(rmses_list)-0.01, max(rmses_list)+0.01, 100)
ax4.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7, label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}', zorder=2)
ax4.set_title('Test R² vs Test RMSE (各Seed综合分布)', fontsize=20, fontweight='bold')
ax4.set_xlabel('Test RMSE', fontsize=18, fontweight='bold')
ax4.set_ylabel('Test R²', fontsize=18, fontweight='bold')
ax4.tick_params(axis='both', labelsize=14)
ax4.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax4.legend(loc='best', prop={'family': 'Times New Roman', 'size': 18})
apply_dual_font_to_fig(fig4)
plt.tight_layout()
plt.savefig(os.path.join(multi_seed_dir, "RF_MultiSeed_scatter.png"), dpi=300)
plt.close(fig4)

print(f"\n模型多Seed稳定性验证报告与可视化生成完毕！图表存入 {multi_seed_dir}")
