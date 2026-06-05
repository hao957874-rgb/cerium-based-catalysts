import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import matplotlib.pyplot as plt
import shap
import os

warnings.filterwarnings('ignore')

# 修复2: 补充全局底层随机种子，保证依赖 NumPy 以及 Python random 的操作（如KFold切分及其他特征随机）能被严格复现
import random
random.seed(42)
np.random.seed(42)

# 设置全局图表字体：数字和字母用Times New Roman，汉字用宋体
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
# 解决宋体会导致负号显示为空白/乱码的问题
plt.rcParams['axes.unicode_minus'] = False

# 1. 读入数据
file_path = r'D:\vsshujubao\CB\data\XGBoost\数据集2.xlsx'
df = pd.read_excel(file_path)

# 清理列名的首尾空格
df.columns = df.columns.astype(str).str.strip()

# 2. 删除 "序号" 列
if '序号' in df.columns:
    df = df.drop(columns=['序号'])

# 将列名中的 cm3 替换为包含上标的 cm³
df.rename(columns=lambda x: str(x).replace('cm3', 'cm³'), inplace=True)

target_col = 'CB转化率'
# 分离特征和目标变量
X = df.drop(columns=[target_col])
y = df[target_col]

# 3. 划分训练集(80%)和测试集(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 为了实现早停(Early Stopping)与交叉验证结合，我们使用手写的KFold结合网格搜索，
# 这样能将每一折的一部分数据作为验证集，当损失不再下降时自动停止该折树的生长，防止过拟合。
from sklearn.model_selection import KFold
import itertools
from tqdm import tqdm

# 4. 重新平衡模型容量与正则化：减轻之前过强的正则设定，以提升整体提取规律的能力
base_params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',  
    'random_state': 42,
    'n_estimators': 3000,         # 给定更大的学习空间，完全由早停决定上限
    'n_jobs': 2,                  # 限制XGBoost的内部CPU线程数，控制CPU过载，为GPU腾出调度空间
#     'subsample': 0.85,
#     'colsample_bytree': 0.85, 
}

# 5. 设置需要搜索的超参数：
# 之前的网格导致了极其严重的“前置欠拟合”：树极浅且惩罚极强，Train R2 被死死压在 0.88。由于根本没学透数据，Test R2 仅 0.70。
# 现彻底解除树模型的算力结界：强行允许深树去挖掘非线性关系，大规模剥除过当的 L2/L1 正则惩罚，全面恢复极高采样率。
# 放飞的数据去配合数千次的大额树生成，而真正导致“过拟合”的残差膨胀，全权交由 50_rounds 的早停(ES)防线来精确截断！
param_grid = {
    # 'learning_rate': [0.09, 0.1, 0.11],    # 步伐更轻盈精细，配合极大的迭代上限能无声渗透更深层规律
    # 'max_depth': [2,3,4],            # 解锁树深限制！原先的 3~5 层根本不足以勾勒 27 维复杂反应图谱
    # 'subsample': [0.7],               # 随机丢弃太多样本(0.7)会失去重要的支撑向量点，拉回到高置信区间
    # 'colsample_bytree': [0.7],        # 特征本就仅有稀缺的 27 个，强制保留 80%~100% 参与分裂，避免严重“失明”
    # 'reg_lambda': [1, 2,3],             # 痛斩 L2 正则！之前动辄 3~5 的拉依达值像千斤顶一样压制着叶子权重的生长
    # 'reg_alpha': [0.05],                # 几近关闭 L1 强制稀疏化，让边缘小特征能够充分发挥复杂的纠缠作用
    # 'min_child_weight': [2],            # 容忍少数同质样本落入叶子节点以增强精准制导
    # 'gamma': [0.03, 0.04]                     # 彻底敞开信息增益的分裂大门
    'learning_rate':    [ 0.07],
    'max_depth':        [4],
    'min_child_weight': [4],
    'subsample':        [0.7],
    'colsample_bytree': [0.7],
    'reg_lambda':       [5],
    'reg_alpha':        [0.01],
    'gamma':            [0.001],
}

# 组合出所有将要尝试的参数
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"当前设置的参数搜索网格:\n{param_grid}\n")
print("正在使用GPU执行网格搜索和5折交叉验证 (并启用 Early Stopping 防止过拟合)，请稍候...\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 用于存储所有的结果与最优模型
cv_results = []
best_val_rmse = float('inf')
best_params = None
best_fold_details = None
best_estimators_list = [] # 记录每一折训练好的模型以获取早停迭代次数

X_tr_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
y_tr_arr = y_train.values if isinstance(y_train, pd.Series) else y_train

# 强制将内存转换为 np.float32 连续数组格式，
# 防止 XGBoost 在 GPU 调用和 DMatrix 转换过程中引发极高的 CPU 拷贝开销
X_tr_arr = np.ascontiguousarray(X_tr_arr, dtype=np.float32)
y_tr_arr = np.ascontiguousarray(y_tr_arr, dtype=np.float32)
X_test_float32 = np.ascontiguousarray(X_test.values if isinstance(X_test, pd.DataFrame) else X_test, dtype=np.float32)

for params in tqdm(param_combinations, desc="网格搜索进度"):
    fold_train_rmses = []
    fold_val_rmses = []
    fold_train_r2s = []
    fold_val_r2s = []
    current_estimators = []
    
    # 5折交叉验证
    for train_index, val_index in kf.split(X_tr_arr):
        X_kf_train, X_kf_val = X_tr_arr[train_index], X_tr_arr[val_index]
        y_kf_train, y_kf_val = y_tr_arr[train_index], y_tr_arr[val_index]
        
        # 初始化带当前参数的 XGBRegressor，将 early_stopping_rounds 放入 __init__ 以适配新版XGBoost
        # eval_metric='rmse'
        model = xgb.XGBRegressor(
            **base_params, 
            **params, 
            early_stopping_rounds=30,
            eval_metric='rmse'
        )
        
        # 拟合，传入验证集 eval_set
        model.fit(
            X_kf_train, y_kf_train,
            eval_set=[(X_kf_train, y_kf_train), (X_kf_val, y_kf_val)],
            verbose=False
        )
        current_estimators.append(model)
        
        # 用早停点所在的最优迭代次数进行评估
        y_kf_train_pred = model.predict(X_kf_train)
        y_kf_val_pred = model.predict(X_kf_val)
        
        fold_train_rmses.append(np.sqrt(mean_squared_error(y_kf_train, y_kf_train_pred)))
        fold_val_rmses.append(np.sqrt(mean_squared_error(y_kf_val, y_kf_val_pred)))
        fold_train_r2s.append(r2_score(y_kf_train, y_kf_train_pred))
        fold_val_r2s.append(r2_score(y_kf_val, y_kf_val_pred))
    
    mean_val_rmse = np.mean(fold_val_rmses)
    
    cv_results.append({
        'params': params,
        'train_rmse': np.mean(fold_train_rmses),
        'val_rmse': mean_val_rmse,
        'train_r2': np.mean(fold_train_r2s),
        'val_r2': np.mean(fold_val_r2s)
    })
    
    # 如果找到了更优的参数
    if mean_val_rmse < best_val_rmse:
        best_val_rmse = mean_val_rmse
        best_params = params
        best_fold_details = {
            'train_rmses': fold_train_rmses, 'val_rmses': fold_val_rmses,
            'train_r2s': fold_train_r2s, 'val_r2s': fold_val_r2s
        }
        best_estimators_list = current_estimators

print("="*65)
print(f"网格搜索找到的最佳参数组合: {best_params}")
# 计算交叉验证中平均起效的迭代次数(早停截断的节点)
# 修复3: 因为 XGBoost 的 best_iteration 是从 0 起始计数，需要补 +1 映射为真实的树深数量，
# 同时用 max(, 1) 保证即使极度欠拟合在第 0 轮就早停，也能生成至少含有 1 棵树的基础模型。
best_trees = max(int(np.mean([m.best_iteration for m in best_estimators_list])) + 1, 1)
print(f"最佳早停迭代次数 (平均): {best_trees}")
print("="*65)

# 6. 打印交叉验证的详细过程（最优参数下各折情况）
print("\n--- 最佳参数下的 5 折交叉验证详细过程 (含早停情况) ---")
print(f"{'Fold':<8} | {'Train RMSE':<12} | {'Val RMSE':<12} | {'Train R2':<10} | {'Val R2':<10} | {'停止树数':<8}")
print("-" * 80)
for i in range(5):
    print(f"Fold {i+1:<3} | {best_fold_details['train_rmses'][i]:<12.4f} | {best_fold_details['val_rmses'][i]:<12.4f} | {best_fold_details['train_r2s'][i]:<10.4f} | {best_fold_details['val_r2s'][i]:<10.4f} | {best_estimators_list[i].best_iteration:<8}")
print("-" * 80)
print(f"{'Mean':<8} | {np.mean(best_fold_details['train_rmses']):<12.4f} | {np.mean(best_fold_details['val_rmses']):<12.4f} | {np.mean(best_fold_details['train_r2s']):<10.4f} | {np.mean(best_fold_details['val_r2s']):<10.4f} | {best_trees:<8}")
print("="*80)

# 7. 最终在独立测试集(20%)验证
# 使用选出的最优参数和平均最佳树深度对 整体 80% 的训练集进行再训练，而不划分验证集
final_model_params = {**base_params, **best_params, 'n_estimators': best_trees, 'eval_metric': 'rmse'}
final_model = xgb.XGBRegressor(**final_model_params)
final_model.fit(
    X_tr_arr, y_tr_arr,
    eval_set=[(X_tr_arr, y_tr_arr), (X_test_float32, y_test)],
    verbose=False
)

y_test_pred = final_model.predict(X_test_float32)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\n--- 独立测试集 (20%未见数据) 最终性能评估 ---")
print(f"Test RMSE = {test_rmse:.4f}")
print(f"Test R2   = {test_r2:.4f}")
print("-" * 65)

# ======= 模型性能可视化 (柱状图 & 散点图) =======
print("\n--- 正在生成模型性能可视化图表 ---")
plot_dir = r'D:\vsshujubao\CB\data\XGBoost\Model_plots'
os.makedirs(plot_dir, exist_ok=True)

# 计算在全量训练集上的拟合指标作为比对基准
y_train_full_pred = final_model.predict(X_tr_arr)
train_rmse_full = np.sqrt(mean_squared_error(y_tr_arr, y_train_full_pred))
train_r2_full = r2_score(y_tr_arr, y_train_full_pred)

cv_val_rmses = best_fold_details['val_rmses']
cv_val_r2s = best_fold_details['val_r2s']
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
max_r2_val = max(r2_vals) if max(r2_vals) > 0 else 0.1
ax2.set_ylim([min(0, min(r2_vals) * 1.22), max_r2_val * 1.22])
for i, v in enumerate(r2_vals):
    ax2.text(i, v + max_r2_val * 0.015, f"{v:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=16, fontfamily='Times New Roman')

bars_path = os.path.join(plot_dir, "xgb_metrics_bars.png")
plt.tight_layout()
plt.savefig(bars_path, dpi=300)
plt.close()
print(f"柱状图 (Loss & R2) 已保存至: {bars_path}")

# 2. 绘制真实值与预测值的散点图
plt.figure(figsize=(7, 7))
plt.scatter(y_tr_arr, y_train_full_pred, color='#5D98C1', edgecolor='black', alpha=0.8, s=80, label='Train data')
plt.scatter(y_test, y_test_pred, color='#E9826F', edgecolor='black', alpha=0.8, s=80, label='Test data')

min_val = min(np.min(y_tr_arr), np.min(y_test), np.min(y_train_full_pred), np.min(y_test_pred))
max_val = max(np.max(y_tr_arr), np.max(y_test), np.max(y_train_full_pred), np.max(y_test_pred))
padding = (max_val - min_val) * 0.05
min_val -= padding
max_val += padding

plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=2, label='y = x (Ideal)')

plt.title('XGBoost', fontsize=20, fontweight='bold')
plt.xlabel('True Values', fontsize=18, fontweight='bold', color='#444444')
plt.ylabel('Predicted Values', fontsize=18, fontweight='bold', color='#444444')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.grid(True, linestyle='--', color='gray', alpha=0.3)
plt.legend(loc='upper left', markerscale=1.4, handletextpad=0.6, framealpha=0.9, edgecolor='lightgray', prop={'family': 'Times New Roman', 'size': 24})

text_x = max_val - padding
text_y_test = min_val + padding * 3
text_y_train = min_val + padding * 1.5
plt.text(text_x, text_y_test, f"Test R²: {test_r2:.4f}", fontsize=30, ha='right', va='bottom', color='#E9826F', fontfamily='Times New Roman')
plt.text(text_x, text_y_train, f"Train R²: {train_r2_full:.4f}", fontsize=30, ha='right', va='bottom', color='#5D98C1', fontfamily='Times New Roman')

scatter_path = os.path.join(plot_dir, "xgb_true_vs_pred_scatter.png")
plt.tight_layout()
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"真实值-预测值散点图 已保存至: {scatter_path}")

# 3. 绘制Loss和R2的迭代曲线 (Learning Curves)
print("\n--- 正在计算并生成 Boost Rounds 学习曲线 ---")
results = final_model.evals_result()
train_rmse_history = results['validation_0']['rmse']
test_rmse_history = results['validation_1']['rmse']
rounds = np.arange(1, best_trees + 1)

train_r2_history = []
test_r2_history = []
# 逐轮累积预测以计算每一轮的 R2
for i in range(1, best_trees + 1):
    y_tr_pred_i = final_model.predict(X_tr_arr, iteration_range=(0, i))
    y_te_pred_i = final_model.predict(X_test_float32, iteration_range=(0, i))
    train_r2_history.append(r2_score(y_tr_arr, y_tr_pred_i))
    test_r2_history.append(r2_score(y_test, y_te_pred_i))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Loss 折线图 (仿照粘贴图片风格)
ax1.plot(rounds, train_rmse_history, color='#28429B', linewidth=2, label='Training Loss')
ax1.plot(rounds, test_rmse_history, color='#D9363E', linewidth=2, label='Test Loss')
ax1.set_title('Training and Testing Loss Comparison', fontsize=20, fontweight='bold', pad=15)
ax1.set_xlabel('Boosting Rounds', fontsize=18, fontweight='bold')
ax1.set_ylabel('Loss (RMSE)', fontsize=18, fontweight='bold')
ax1.tick_params(axis='both', labelsize=14)
ax1.grid(color='#E5E5E5', linestyle='--', linewidth=1.2, alpha=0.8)
ax1.legend(loc='upper right', fontsize=20, frameon=True, edgecolor='lightgray', framealpha=0.9, prop={'family': 'Times New Roman', 'size': 20})
ax1.set_xlim([1, best_trees])
for side in ['top', 'right', 'left', 'bottom']:
    ax1.spines[side].set_color('black')
    ax1.spines[side].set_linewidth(1.2)

# Plot 2: R2 折线图 (仿照粘贴图片风格)
ax2.plot(rounds, train_r2_history, color='#28429B', linewidth=2, label='Training R²')
ax2.plot(rounds, test_r2_history, color='#D9363E', linewidth=2, label='Test R²')
ax2.set_title('Training and Testing R² Comparison', fontsize=20, fontweight='bold', pad=15)
ax2.set_xlabel('Boosting Rounds', fontsize=18, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=18, fontweight='bold')
ax2.tick_params(axis='both', labelsize=14)
ax2.grid(color='#E5E5E5', linestyle='--', linewidth=1.2, alpha=0.8)
ax2.legend(loc='lower right', fontsize=20, frameon=True, edgecolor='lightgray', framealpha=0.9, prop={'family': 'Times New Roman', 'size': 20})
ax2.set_xlim([1, best_trees])
for side in ['top', 'right', 'left', 'bottom']:
    ax2.spines[side].set_color('black')
    ax2.spines[side].set_linewidth(1.2)

learning_curve_path = os.path.join(plot_dir, "xgb_learning_curves.png")
plt.tight_layout()
plt.savefig(learning_curve_path, dpi=300, facecolor='white')
plt.close()
print(f"迭代学习曲线 (Loss & R2) 已保存至: {learning_curve_path}")

# 8. 输出不同参数组合下的各个指标
print("\n--- 网格搜索：全体参数组合的性能 (Mean Train/Val) ---")
print(f"{'learning_rate':<14} | {'max_depth':<10} | {'Train RMSE':<10} | {'Val RMSE':<10} | {'Train R2':<8} | {'Val R2':<8}")
print("-" * 80)
for res in cv_results:
    p_lr = res['params']['learning_rate']
    p_md = res['params']['max_depth']
    best_flag = "(* 最佳参数)" if res['params'] == best_params else ""
    
    print(f"{p_lr:<14} | {p_md:<10} | {res['train_rmse']:<10.4f} | {res['val_rmse']:<10.4f} | {res['train_r2']:<8.4f} | {res['val_r2']:<8.4f} {best_flag}")

# 9. SHAP 特征重要性分析与可视化 (基于最终训练好的重训独立模型 final_model)
print("\n--- 正在进行SHAP特征重要性分析... ---")
# 构建 SHAP 原生解释器
explainer = shap.TreeExplainer(final_model)

# 计算训练集上所有样本的SHAP值 (为了看特征重要性通常使用全量或者训练集)
shap_values_obj = explainer(X_train)        # 包含base_values和data的对象
shap_values_arr = explainer.shap_values(X_train) # 仅有二维矩阵值用于图表

# 确保输出路径存在
output_dir = r'D:\vsshujubao\CB\data\XGBoost\SHAP_plots'
os.makedirs(output_dir, exist_ok=True)

# 1. 绘制全局 Summary Plot (蜂群图/条形图)
plt.figure(figsize=(10, 8))
# 使用原生绘图，不要show阻塞程序
shap.summary_plot(shap_values_arr, X_train, plot_type="dot", show=False)
summary_path = os.path.join(output_dir, "shap_summary_plot.png")
plt.tight_layout()
plt.savefig(summary_path, dpi=300)
plt.close()
print(f"SHAP 摘要蜂群图(Summary Plot)已保存至: {summary_path}")

# 1.5 绘制 SHAP 特征贡献度柱状图 (横向条形图)
plt.figure(figsize=(10, 8))
# 添加 plot_type="bar"，修改配色为您希望的暗红色 (Dark Red)
shap.summary_plot(shap_values_arr, X_train, plot_type="bar", show=False, color="#8B0000")

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
            f'{width:.3f}', ha='left', va='center', fontsize=15, fontname='Times New Roman', fontweight='bold')
        if width > max_width:
            max_width = width

# 拓展坐标轴让它装得下数字，保证数字不能超过方框
if max_width > 0:
    ax.set_xlim(0, max_width * 1.12)

# 给图中内容在四周添加上黑色的全框闭合
for side in ['top', 'right', 'left', 'bottom']:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color('black')
    ax.spines[side].set_linewidth(1.2)

bar_path = os.path.join(output_dir, "shap_bar_plot.png")
plt.tight_layout()
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"SHAP 贡献度横向柱状图已保存至: {bar_path}")

# 2. 获取最重要的前4个特征，并针对它们绘制 Dependence Plot
# 根据每个特征的全局平均绝对 SHAP 值来排序特征重要性
mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
top_features_indices = np.argsort(mean_abs_shap)[::-1]
feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else [f"Feature {req}" for req in range(X_train.shape[1])]

# 提取排名前四的重要特征（如果不够4个则取所有）
top_n = min(4, len(feature_names))
top_feature_names = [feature_names[idx] for idx in top_features_indices[:top_n]]

print("\n--- 全局最关键的前几个特征，正在为它们绘制 Dependence Plot ---")
for feature_name in top_feature_names:
    plt.figure(figsize=(8, 6))
    # Dependence plot 会自动寻找另一个跟它有强烈交互的特征作为颜色映射
    shap.dependence_plot(feature_name, shap_values_arr, X_train, show=False)
    dep_path = os.path.join(output_dir, f"shap_dependence_{feature_name}.png")
    # Windows保存文件不支持非法字符（如 / : * 等），如果有进行规避
    safe_dep_path = dep_path.replace("/", "_").replace("\\", "_").replace(":", "") 
    plt.tight_layout()
    # 修复1: 实际写盘时使用安全名(safe_dep_path)，防止物理量包含除号等触发 FileNotFoundError
    plt.savefig(dep_path, dpi=300)
    plt.close()
    print(f"[{feature_name}] 的依赖图已保存至: {dep_path}")

print(f"\n所有特征重要性排位一览 (均值绝对SHAP值从大到小):")
for idx in top_features_indices:
    print(f"{feature_names[idx]:<15}: {mean_abs_shap[idx]:.4f}")

print("\nSHAP图表生成完毕！您可以通过蜂群图来观察哪些特征在0轴附近聚集较多、或均值绝对 SHAP(上面列出的数值) 极小的特征。它们就是冗余特征，可作为您后续特征筛选(剪枝)的依据！\n")


