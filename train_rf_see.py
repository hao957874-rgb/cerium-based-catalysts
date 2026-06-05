import pandas as pd
import numpy as np
import os
import re
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 设置全局图表字体：数字和字母用Times New Roman，汉字用宋体
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
# 解决宋体会导致负号显示为空白/乱码的问题
plt.rcParams['axes.unicode_minus'] = False


# 1. 设置工作状态与读取数据
print("="*65)
file_path = r'D:\vsshujubao\CB\data\RF\数据集2.xlsx'
print(f"正在读取数据文件: {file_path}")

df = pd.read_excel(file_path)

if '序号' in df.columns:
    df = df.drop(columns=['序号'])

# 将列名中的 cm3 替换为包含上标的 cm³
df.rename(columns=lambda x: str(x).replace('cm3', 'cm³'), inplace=True)

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
    # "n_estim
    # "max_depth": [10, 12, 14],
    # "min_samples_split":
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

print(f"数据处理完毕。训练集规模：{X_train_arr.shape[0]}")
print("-" * 65)
print("开始执行 随机森林(RF) 网格搜索...")
print("提示：Scikit-Learn 的 RandomForest 原生主要走 CPU，已自动为您开启全核心并发 (n_jobs=-1) 以最大化速度")
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
# ② 修复:
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
print(f"{'参数配置':<45} | {'Train RMSE':<12} | {'Val RMSE':<12} | {'Overfit Gap':<12} | {'Train R2':<8} | {'Val R2':<8}")
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

# ======= 模型性能可视化 (柱状图 & 散点图 & 学习曲线) =======
print("\n--- 正在生成模型性能可视化图表 ---")
plot_dir = r'D:\vsshujubao\CB\data\RF\Model_plots_RF'
os.makedirs(plot_dir, exist_ok=True)

# 计算在全量训练集上的拟合指标作为对比基准
y_train_full_pred = best_model.predict(X_train_arr)
train_rmse_full = np.sqrt(mean_squared_error(y_train, y_train_full_pred))
train_r2_full = r2_score(y_train, y_train_full_pred)

cv_val_rmses = best_fold_details['val_rmse']
cv_val_r2s = best_fold_details['val_r2']
cv_avg_rmse = np.mean(cv_val_rmses)
cv_avg_r2 = np.mean(cv_val_r2s)

# 1. 绘制柱状图 (Loss和R2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
labels = ['Train\n(Full)', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'CV Avg', 'Test']
rmse_vals = [train_rmse_full] + cv_val_rmses + [cv_avg_rmse, test_rmse]
r2_vals = [train_r2_full] + cv_val_r2s + [cv_avg_r2, test_r2]

colors = ['#5D98C1'] + ['#F4D03F']*5 + ['#D4AF37', '#E9826F']
bar_width = 0.5

ax1.bar(labels, rmse_vals, color=colors, edgecolor='black', alpha=0.8, width=bar_width)
ax1.set_title('Loss (RMSE) Comparison', fontsize=20, fontweight='bold')
ax1.set_ylabel('RMSE Score', fontsize=18, fontweight='bold')
ax1.set_xticklabels(labels, fontsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.tick_params(axis='x', rotation=0)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.set_ylim(0, max(rmse_vals) * 1.22)
for i, v in enumerate(rmse_vals):
    ax1.text(i, v + np.max(rmse_vals)*0.015, f"{v:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=16, fontfamily='Times New Roman')

ax2.bar(labels, r2_vals, color=colors, edgecolor='black', alpha=0.8, width=bar_width)
ax2.set_title('R² Comparison', fontsize=20, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=18, fontweight='bold')
ax2.set_xticklabels(labels, fontsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='x', rotation=0)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
max_r2 = max(r2_vals) if max(r2_vals) > 0 else 0.1
ax2.set_ylim([min(0, min(r2_vals)*1.22), max_r2 * 1.22])
for i, v in enumerate(r2_vals):
    ax2.text(i, v + max_r2 * 0.015, f"{v:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=16, fontfamily='Times New Roman')

bars_path = os.path.join(plot_dir, "rf_metrics_bars.png")
plt.tight_layout()
plt.savefig(bars_path, dpi=300)
plt.close()
print(f"柱状图 (Loss & R2) 已保存至: {bars_path}")

# 2. 绘制真实值与预测值的散点图 (仿照SVR风格)
plt.figure(figsize=(7, 7))
plt.scatter(y_train, y_train_full_pred, color='#74a9cf', edgecolor='black', alpha=0.8, s=80, label='Train data')
plt.scatter(y_test, test_pred, color='#e68a83', edgecolor='black', alpha=0.8, s=80, label='Test data')

min_val = min(np.min(y_train), np.min(y_test), np.min(y_train_full_pred), np.min(test_pred))
max_val = max(np.max(y_train), np.max(y_test), np.max(y_train_full_pred), np.max(test_pred))
padding = (max_val - min_val) * 0.05
min_val -= padding
max_val += padding

plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=2, label='y = x (Ideal)')

plt.title('Random Forest', fontsize=20, fontweight='bold')
plt.xlabel('True Values', fontsize=18, fontweight='bold', color='#444444')
plt.ylabel('Predicted Values', fontsize=18, fontweight='bold', color='#444444')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle='--', color='lightgray', alpha=0.6)
plt.legend(loc='upper left', markerscale=1.4, handletextpad=0.6, framealpha=0.9, edgecolor='lightgray', prop={'family': 'Times New Roman', 'size': 24})

text_x = max_val - padding
text_y_test = min_val + padding * 3
text_y_train = min_val + padding * 1.5
plt.text(text_x, text_y_test, f"Test R²: {test_r2:.4f}", fontsize=30, ha='right', va='bottom', color='#e68a83', fontfamily='Times New Roman')
plt.text(text_x, text_y_train, f"Train R²: {train_r2_full:.4f}", fontsize=30, ha='right', va='bottom', color='#74a9cf', fontfamily='Times New Roman')

# 为散点图添加四周黑框
for side in ['top', 'right', 'left', 'bottom']:
    plt.gca().spines[side].set_color('black')
    plt.gca().spines[side].set_linewidth(1.2)

scatter_path = os.path.join(plot_dir, "rf_true_vs_pred_scatter.png")
plt.tight_layout()
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"真实值-预测值散点图 已保存至: {scatter_path}")

# 3. 绘制Loss和R2的学习曲线 (随建树数量的变化)
print("\n--- 正在计算并生成 Number of Trees 学习曲线 ---")
n_trees = best_model.n_estimators
train_preds_sum = np.zeros(len(X_train_arr))
test_preds_sum = np.zeros(len(X_test_arr))

train_rmse_history = []
test_rmse_history = []
train_r2_history = []
test_r2_history = []

for i, tree in enumerate(best_model.estimators_):
    train_preds_sum += tree.predict(X_train_arr)
    test_preds_sum += tree.predict(X_test_arr)
    
    curr_train_pred = train_preds_sum / (i + 1)
    curr_test_pred = test_preds_sum / (i + 1)
    
    train_rmse_history.append(np.sqrt(mean_squared_error(y_train, curr_train_pred)))
    test_rmse_history.append(np.sqrt(mean_squared_error(y_test, curr_test_pred)))
    train_r2_history.append(r2_score(y_train, curr_train_pred))
    test_r2_history.append(r2_score(y_test, curr_test_pred))

rounds = np.arange(1, n_trees + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Loss 折线图 (仿照粘贴图片风格)
ax1.plot(rounds, train_rmse_history, color='#28429B', linewidth=2, label='Training Loss')
ax1.plot(rounds, test_rmse_history, color='#D9363E', linewidth=2, label='Test Loss')
ax1.set_title('Training and Testing Loss Comparison', fontsize=20, fontweight='bold', pad=15)
ax1.set_xlabel('Number of Trees', fontsize=18, fontweight='bold')
ax1.set_ylabel('Loss (RMSE)', fontsize=18, fontweight='bold')
ax1.tick_params(axis='both', labelsize=14)
ax1.grid(color='#E5E5E5', linestyle='--', linewidth=1.2, alpha=0.8)
ax1.legend(loc='upper right', fontsize=20, frameon=True, edgecolor='lightgray', framealpha=0.9, prop={'family': 'Times New Roman', 'size': 20})
ax1.set_xlim([1, n_trees])
for side in ['top', 'right', 'left', 'bottom']:
    ax1.spines[side].set_color('black')
    ax1.spines[side].set_linewidth(1.2)

# Plot 2: R2 折线图 (仿照粘贴图片风格)
ax2.plot(rounds, train_r2_history, color='#28429B', linewidth=2, label='Training R²')
ax2.plot(rounds, test_r2_history, color='#D9363E', linewidth=2, label='Test R²')
ax2.set_title('Training and Testing R² Comparison', fontsize=20, fontweight='bold', pad=15)
ax2.set_xlabel('Number of Trees', fontsize=18, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=18, fontweight='bold')
ax2.tick_params(axis='both', labelsize=14)
ax2.grid(color='#E5E5E5', linestyle='--', linewidth=1.2, alpha=0.8)
ax2.legend(loc='lower right', fontsize=20, frameon=True, edgecolor='lightgray', framealpha=0.9, prop={'family': 'Times New Roman', 'size': 20})
ax2.set_xlim([1, n_trees])
for side in ['top', 'right', 'left', 'bottom']:
    ax2.spines[side].set_color('black')
    ax2.spines[side].set_linewidth(1.2)

learning_curve_path = os.path.join(plot_dir, "rf_learning_curves.png")
plt.tight_layout()
plt.savefig(learning_curve_path, dpi=300, facecolor='white')
plt.close()
print(f"树数量增加的学习曲线 (Loss & R2) 已保存至: {learning_curve_path}")


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

# (1) 绘制 Summary Plot (摘要图/蜂群图)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_arr, X_test_arr, feature_names=feature_names, show=False)
summary_path = os.path.join(output_dir, "rf_shap_summary_plot.png")
plt.tight_layout()
plt.savefig(summary_path, dpi=300)
plt.close()
print(f"\nSHAP 摘要蜂群图 (Summary Plot) 已保存至:\n  -> {summary_path}")

# (1.5) 绘制 SHAP 特征贡献度柱状图 (横向条形图)
plt.figure(figsize=(10, 8))
# 添加 plot_type="bar" 并且使用暗红色 (#8B0000)
shap.summary_plot(shap_values_arr, X_test_arr, feature_names=feature_names, plot_type="bar", show=False, color="#8B0000")

ax = plt.gca()
# 删除图下面自带的 (average impact on model output magnitude) 这个内容
ax.set_xlabel("")
# 统一标题与刻度字号
ax.set_title("SHAP Feature Contributions", fontsize=20, fontweight='bold', pad=15)
ax.tick_params(axis='both', labelsize=16)

max_width = 0
# 给图中每一个柱子标上具体的数值
for p in ax.patches:
    width = p.get_width()
    if width > 0: # 避免标注空柱或负边
        # 加上具体的数值，稍微偏右一点点
        ax.text(width * 1.02, p.get_y() + p.get_height() / 2, 
            f'{width:.5f}', ha='left', va='center', fontsize=15, fontname='Times New Roman', fontweight='bold')
        if width > max_width:
            max_width = width

# 拓展坐标轴让它装得下数字，保证数字不能超过方框
if max_width > 0:
    ax.set_xlim(0, max_width * 1.15)

# 给图中内容在四周添加上黑色的全框闭合
for side in ['top', 'right', 'left', 'bottom']:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color('black')
    ax.spines[side].set_linewidth(1.2)

bar_path = os.path.join(output_dir, "rf_shap_bar_plot.png")
plt.tight_layout()
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"SHAP 贡献度横向柱状图已保存至:\n  -> {bar_path}")


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
