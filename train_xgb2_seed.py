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

# 设置中文字体以及负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

# 1. 读入数据
file_path = r'D:\vsshujubao\CB\data\XGBoost\数据集2.xlsx'
df = pd.read_excel(file_path)

# 清理列名的首尾空格
df.columns = df.columns.astype(str).str.strip()

# 2. 删除 "序号" 列
if '序号' in df.columns:
    df = df.drop(columns=['序号'])

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
final_model_params = {**base_params, **best_params, 'n_estimators': best_trees}
final_model = xgb.XGBRegressor(**final_model_params)
final_model.fit(X_tr_arr, y_tr_arr)

y_test_pred = final_model.predict(X_test_float32)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\n--- 独立测试集 (20%未见数据) 最终性能评估 ---")
print(f"Test RMSE = {test_rmse:.4f}")
print(f"Test R2   = {test_r2:.4f}")
print("-" * 65)

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
    plt.savefig(safe_dep_path, dpi=300)
    plt.close()
    print(f"[{feature_name}] 的依赖图已保存至: {safe_dep_path}")

print(f"\n所有特征重要性排位一览 (均值绝对SHAP值从大到小):")
for idx in top_features_indices:
    print(f"{feature_names[idx]:<15}: {mean_abs_shap[idx]:.4f}")

print("\nSHAP图表生成完毕！您可以通过蜂群图来观察哪些特征在0轴附近聚集较多、或均值绝对 SHAP(上面列出的数值) 极小的特征。它们就是冗余特征，可作为您后续特征筛选(剪枝)的依据！\n")

# ==============================================================================
# --- 新增模块：多 Seed 稳定性验证阶段 ---
# ==============================================================================
print("\n" + "="*80)
print(">>> 启动多 Seed 稳定性验证模块")
print("================================================================================")

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42, 20, 30, 50, 123, 256, 512, 1024, 2048, 4096]
multi_seed_results = []

import logging
import seaborn as sns
# 禁用 XGBoost 内部不必要的告警信息
logging.getLogger("xgboost").setLevel(logging.ERROR)

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
    # 1. 各 seed 控制数据集的划分
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # 转换为 np.float32 连续数组
    X_tr_arr_seed = np.ascontiguousarray(X_tr.values if isinstance(X_tr, pd.DataFrame) else X_tr, dtype=np.float32)
    y_tr_arr_seed = np.ascontiguousarray(y_tr.values if isinstance(y_tr, pd.Series) else y_tr, dtype=np.float32)
    X_te_arr_seed = np.ascontiguousarray(X_te.values if isinstance(X_te, pd.DataFrame) else X_te, dtype=np.float32)
    
    best_val_rmse_seed = float('inf')
    best_params_seed = None
    best_trees_seed = 1
    
    # 用当前seed创建独立的KFold
    kf_seed = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    # 覆盖 base_params 中的 random_state
    base_params_seed = {**base_params, 'random_state': seed}
    
    # 2. 网格搜索与交叉验证 (基于外层的 param_combinations 与恒定的 KFold 配置)
    for params in param_combinations:
        fold_val_rmses = []
        current_estimators = []
        for train_idx, val_idx in kf_seed.split(X_tr_arr_seed):
            X_kf_train, X_kf_val = X_tr_arr_seed[train_idx], X_tr_arr_seed[val_idx]
            y_kf_train, y_kf_val = y_tr_arr_seed[train_idx], y_tr_arr_seed[val_idx]
            
            model = xgb.XGBRegressor(
                **base_params_seed, 
                **params, 
                early_stopping_rounds=30,
                eval_metric='rmse'
            )
            model.fit(
                X_kf_train, y_kf_train,
                eval_set=[(X_kf_train, y_kf_train), (X_kf_val, y_kf_val)],
                verbose=False
            )
            current_estimators.append(model)
            y_kf_val_pred = model.predict(X_kf_val)
            fold_val_rmses.append(np.sqrt(mean_squared_error(y_kf_val, y_kf_val_pred)))
        
        mean_val_rmse = np.mean(fold_val_rmses)
        if mean_val_rmse < best_val_rmse_seed:
            best_val_rmse_seed = mean_val_rmse
            best_params_seed = params
            best_trees_seed = max(int(np.mean([m.best_iteration for m in current_estimators])) + 1, 1)
            
    # 3. 在独立测试集 (20%) 验证最终模型表现
    final_model_seed_params = {**base_params_seed, **best_params_seed, 'n_estimators': best_trees_seed}
    final_model_seed = xgb.XGBRegressor(**final_model_seed_params)
    final_model_seed.fit(X_tr_arr_seed, y_tr_arr_seed)
    
    y_te_pred = final_model_seed.predict(X_te_arr_seed)
    seed_rmse = np.sqrt(mean_squared_error(y_te, y_te_pred))
    seed_r2 = r2_score(y_te, y_te_pred)
    
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

print("\n================================================================================")
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
# 绘制四大可视化图
multi_seed_dir = r'D:\vsshujubao\CB\data\XGBoost\MultiSeed_plots'
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
plt.savefig(os.path.join(multi_seed_dir, "Fig1_Test_R2_Bar.png"), dpi=300)
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
plt.savefig(os.path.join(multi_seed_dir, "Fig2_Test_RMSE_Bar.png"), dpi=300)
plt.close(fig2)

# 图3：Test R2 箱线图 + 散点
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.boxplot(y=r2s_list, ax=ax3, color='lightgray', showfliers=False, width=0.4, zorder=1)
sns.stripplot(y=r2s_list, ax=ax3, color='#E58066', size=8, jitter=True, alpha=0.8, edgecolor='black', linewidth=1, zorder=2)
ax3.set_title('Test R² 离散分布箱线图 (20 Seeds)', fontsize=20, fontweight='bold')
ax3.set_ylabel('Test R²', fontsize=18, fontweight='bold')
ax3.tick_params(axis='y', labelsize=14)
ax3.grid(axis='y', linestyle='--', alpha=0.5)
apply_dual_font_to_fig(fig3)
plt.tight_layout()
plt.savefig(os.path.join(multi_seed_dir, "Fig3_Test_R2_Boxplot.png"), dpi=300)
plt.close(fig3)

# 图4：Test RMSE vs Test R2 散点图 + 趋势线
fig4, ax4 = plt.subplots(figsize=(10, 6))
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
plt.savefig(os.path.join(multi_seed_dir, "Fig4_RMSE_vs_R2_Scatter.png"), dpi=300)
plt.close(fig4)

print("\n模型多Seed稳定性验证报告与可视化生成完毕！图表存入 D:\\vsshujubao\\CB\\data\\XGBoost\\MultiSeed_plots")


