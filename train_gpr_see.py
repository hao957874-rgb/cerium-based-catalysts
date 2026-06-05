import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
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

# 2. 划分训练集和测试集 (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# 转换为了方便索引提取
X_train_vals = X_train.values
X_test_vals = X_test.values

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

print(f"数据处理完毕。训练集规模：{X_train_vals.shape} | 测试集规模：{X_test_vals.shape}")
print("-" * 65)
print("开始执行 GPR 网格搜索与 5 折交叉验证...")
print("提示：Scikit-Learn 的 GPR 仅支持 CPU 进行运算。如需 GPU 支持，必须配置 GPyTorch 环境。当前自动使用 CPU 执行。")
print(f"搜索空间包含 {len(grid)} 种组合。\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
best_val_rmse = float('inf')
best_params = None
best_fold_details = None
best_model = None

# 为了在显示结果时能清晰表明核函数类型
def get_kernel_name(kernel):
    k_str = str(kernel)
    if "RBF" in k_str: return "RBF"
    if "Matern" in k_str: return "Matern"
    if "RationalQuadratic" in k_str: return "RatQuad"
    return "Unknown"

# 5. 网格搜索进度条
for params in tqdm(grid, desc="GPR网格搜索及5折交叉验证进度"):
    f_tr_rmse, f_val_rmse, f_tr_r2, f_val_r2 = [], [], [], []
    
    # 5折交叉验证
    for train_idx, val_idx in kf.split(X_train_vals):
        # 折内分割
        X_kf_train_raw, X_kf_val_raw = X_train_vals[train_idx], X_train_vals[val_idx]
        y_kf_train, y_kf_val = y_train[train_idx], y_train[val_idx]
        
        # 折内独立拟合 Scaler，并在折内的验证集上进行 Transform，防止信息泄露
        inner_scaler = StandardScaler()
        X_kf_train = inner_scaler.fit_transform(X_kf_train_raw)
        X_kf_val = inner_scaler.transform(X_kf_val_raw)
        
        # 训练模型
        model = GaussianProcessRegressor(
            kernel=clone(params['kernel']),
            alpha=params['alpha'],
            n_restarts_optimizer=params['n_restarts_optimizer'],
            random_state=42,
            normalize_y=True   # 自动对目标值标准化，有助于GPR拟合
        )
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
        'kernel_name': get_kernel_name(params['kernel']),
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
        # 使用当前找到的最佳参数，保留其结构从头构建隔离的最佳模型
        best_model = GaussianProcessRegressor(
            kernel=clone(params['kernel']),
            alpha=params['alpha'],
            n_restarts_optimizer=params['n_restarts_optimizer'],
            random_state=42,
            normalize_y=True
        )

# 8. 独立测试集表现
# 全局标准化（用于最终模型在整个训练集上拟合，随后针对未见过的测试集预测）
final_scaler = StandardScaler()
X_train_final_scaled = final_scaler.fit_transform(X_train_vals)
X_test_final_scaled = final_scaler.transform(X_test_vals)

best_model.fit(X_train_final_scaled, y_train)

# 6. 打印交叉验证明细表 (集中输出)
print("\n" + "="*80)
print(f"网格搜索找到的最佳参数组合:")
print(f"    Kernel: {get_kernel_name(best_params['kernel'])}")
print(f"    Alpha : {best_params['alpha']}")
print(f"    Restarts: {best_params['n_restarts_optimizer']}")
print("="*80)

print("\n--- 最佳参数下的 5 折交叉验证详细过程 ---")
print(f"{'Fold':<8} | {'Train RMSE (Loss)':<17} | {'Val RMSE (Loss)':<17} | {'Train R2':<10} | {'Val R2':<10}")
print("-" * 80)
for i in range(5):
    print(f"Fold {i+1:<3} | {best_fold_details['tr_rmse'][i]:<17.4f} | {best_fold_details['val_rmse'][i]:<17.4f} | {best_fold_details['tr_r2'][i]:<10.4f} | {best_fold_details['val_r2'][i]:<10.4f}")
print("-" * 80)
print(f"{'Mean':<8} | {np.mean(best_fold_details['tr_rmse']):<17.4f} | {np.mean(best_fold_details['val_rmse']):<17.4f} | {np.mean(best_fold_details['tr_r2']):<10.4f} | {np.mean(best_fold_details['val_r2']):<10.4f}")
print("="*80)

# 7. 全网格搜索不同参数下的汇总表
print("\n--- 全网格搜索不同参数下的结果组合 (按 Val RMSE 升序排序) ---")
print(f"{'Params (Kernel, alpha, restart)':<35} | {'Train RMSE':<12} | {'Val RMSE':<12} | {'Train R2':<8} | {'Val R2':<8}")
print("-" * 85)
results_sorted = sorted(results, key=lambda x: x['val_rmse'])
for res in results_sorted:
    p_str = f"{res['kernel_name']}, a={res['params']['alpha']}, r={res['params']['n_restarts_optimizer']}"
    best_flag = "(*Opt)" if res['params'] == best_params else ""
    print(f"{p_str:<35} | {res['train_rmse']:<12.4f} | {res['val_rmse']:<12.4f} | {res['train_r2']:<8.4f} | {res['val_r2']:<8.4f} {best_flag}")

# =================最终表现=================
test_pred = best_model.predict(X_test_final_scaled)
train_pred = best_model.predict(X_train_final_scaled)

test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)
train_rmse_final = np.sqrt(mean_squared_error(y_train, train_pred))
train_r2_final = r2_score(y_train, train_pred)

print("\n" + "="*80)
print("--- 测试集 (20%未见数据) 最终性能评估 ---")
print(f"Test RMSE (Loss) = {test_rmse:.4f}")
print(f"Test R2          = {test_r2:.4f}")
print("="*80)

# 可视化部分
plot_dir = r'D:\vsshujubao\CB\data\GPR\Model_plots_GPR'
os.makedirs(plot_dir, exist_ok=True)

# 解决画图中文乱码及负号显示问题：使用专门的中英文混排字体回退配置
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun', 'STSong', 'Songti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 提取5折数据
cv_rmses = best_fold_details['val_rmse']
cv_r2s = best_fold_details['val_r2']
avg_cv_rmse = np.mean(cv_rmses)
avg_cv_r2 = np.mean(cv_r2s)

labels = ['Train'] + [f'Fold {i+1}' for i in range(5)] + ['CV Avg', 'Test']
rmse_vals = [train_rmse_final] + list(cv_rmses) + [avg_cv_rmse, test_rmse]
r2_vals = [train_r2_final] + list(cv_r2s) + [avg_cv_r2, test_r2]

# 1. 8-Bar 性能柱状图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
bar_width = 0.5
colors_rmse = ['#5D98C1'] + ['#F4D03F']*5 + ['#D4AF37', '#E9826F']
colors_r2 = ['#5D98C1'] + ['#F4D03F']*5 + ['#D4AF37', '#E9826F']

x = np.arange(len(labels))
b1 = ax1.bar(x, rmse_vals, color=colors_rmse, edgecolor='black', linewidth=1.2, width=bar_width)
ax1.set_title('Loss (RMSE) across Training, CV and Test', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
ax1.set_ylabel('RMSE', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=14, fontfamily='Times New Roman')
ax1.tick_params(axis='y', labelsize=14)
ax1.set_ylim(0, max(rmse_vals) * 1.22)
for idx, rect in enumerate(b1):
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., h + max(rmse_vals)*0.015, f'{h:.4f}', ha='center', va='bottom', fontsize=18, fontfamily='Times New Roman', fontweight='bold')
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

b2 = ax2.bar(x, r2_vals, color=colors_r2, edgecolor='black', linewidth=1.2, width=bar_width)
ax2.set_title('R² Score across Training, CV and Test', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
ax2.set_ylabel('R²', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=14, fontfamily='Times New Roman')
ax2.tick_params(axis='y', labelsize=14)
max_r2_val = max(r2_vals) if max(r2_vals) > 0 else 0.1
ax2.set_ylim([min(0, min(r2_vals) * 1.22), max_r2_val * 1.22])
for idx, rect in enumerate(b2):
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h + max_r2_val*0.015, f'{h:.4f}', ha='center', va='bottom', fontsize=18, fontfamily='Times New Roman', fontweight='bold')
for spine in ax2.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "gpr_performance_bars.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2. 散点图
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
plt.title('GPR', fontsize=20, fontweight='bold', fontfamily='Times New Roman')
plt.xlabel('True Values', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('Predicted Values', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
plt.legend(loc='upper left', markerscale=1.4, handletextpad=0.6, prop={'family': 'Times New Roman', 'size': 28})
plt.grid(True, linestyle='--', alpha=0.4)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.text(0.95, 0.05, f'Train R²: {train_r2_final:.4f}', transform=ax.transAxes, fontsize=32, color='#5D98C1', ha='right', fontfamily='Times New Roman')
plt.text(0.95, 0.12, f'Test R²: {test_r2:.4f}', transform=ax.transAxes, fontsize=32, color='#E9826F', ha='right', fontfamily='Times New Roman')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "gpr_scatter_plot.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"评估与图表已保存至: {plot_dir}")

# 9. SHAP 特征重要性分析
print("\n--- 正在进行 SHAP 特征重要性分析 (模型偏大，请耐心等待) ---")
# 对于 GPR 为避免计算时长爆炸，使用 KMeans 初始化背景数据，选取 30 个背景点代表全局 (必须基于用来最终拟合的训练集)
background = shap.kmeans(X_train_final_scaled, 50)
explainer = shap.KernelExplainer(best_model.predict, background)

# 为避免误导，且准确评价模型的泛化特征依赖性，采用真实的测试集进行 SHAP 计算
X_shap_eval_scaled = X_test_final_scaled

# X_vis 还原成 DataFrame 供画图显示列名（使用真正的独立测试集部分的原始量纲数据来画依赖图，业务含义最清晰）
X_vis = X_test.copy()

# 将 tqdm 加入 SHAP 评估阶段
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    shap_values = explainer.shap_values(X_shap_eval_scaled, nsamples=100)

shap_values_arr = np.array(shap_values)

# 配置文件输出路径
output_dir = r'D:\vsshujubao\CB\data\GPR\SHAP_plots_GPR'
os.makedirs(output_dir, exist_ok=True)

# 解决画图中文乱码及负号显示问题
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun', 'STSong', 'FangSong', 'SimHei']
plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun', 'STSong', 'FangSong', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# (0) 绘制自定义 SHAP 柱状图 (深红色, 横向)
plt.figure(figsize=(10, 8))
mean_abs_impact = np.abs(shap_values_arr).mean(axis=0)
sorted_idx = np.argsort(mean_abs_impact)
pos_features = np.array(feature_names)[sorted_idx]
pos_impacts = mean_abs_impact[sorted_idx]

bars = plt.barh(np.arange(len(pos_features)), pos_impacts, color='#8B0000', edgecolor='black', linewidth=1)
# 强制让 Y 轴 (汉字特征名) 使用宋体，避免被其他不支持汉字的字体吞掉
plt.yticks(np.arange(len(pos_features)), pos_features, fontname='SimSun', fontsize=16)
plt.xticks(fontname='Times New Roman', fontsize=16)
plt.xlabel('') # 隐藏 xlabel
plt.title('SHAP Feature Importance', fontname='Times New Roman', fontsize=20, fontweight='bold', pad=15)

max_width = max(pos_impacts)
plt.xlim(0, max_width * 1.15) # 防止文字超出边界
ax = plt.gca()
ax.tick_params(axis='both', labelsize=16)
for i, rect in enumerate(bars):
    width = rect.get_width()
    ax.text(width * 1.02, rect.get_y() + rect.get_height() / 2,
            f'{width:.4f}', ha='left', va='center',
            fontname='Times New Roman', fontsize=15, fontweight='bold')

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.tight_layout()
bar_path = os.path.join(output_dir, "gpr_shap_bar_plot.png")
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSHAP 贡献率柱状图 已保存至:\n  -> {bar_path}")

# (1) 绘制 Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_arr, X_vis, max_display=len(feature_names), show=False)

fig = plt.gcf()
for ax_ in fig.axes:
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

summary_path = os.path.join(output_dir, "gpr_shap_summary_plot.png")
plt.tight_layout()
plt.savefig(summary_path, dpi=300)
plt.close()
print(f"SHAP 摘要蜂群图 (Summary Plot) 已保存至:\n  -> {summary_path}")

# (2) 绘制 最重要特征的 Dependence Plot
mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
sorted_indices = np.argsort(mean_abs_shap)[::-1]
top_feature = feature_names[sorted_indices[0]]

plt.figure(figsize=(8, 6))
shap.dependence_plot(sorted_indices[0], shap_values_arr, X_vis, show=False)

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
