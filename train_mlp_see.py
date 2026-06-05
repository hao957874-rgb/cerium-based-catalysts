import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')

# 根据当前脚本位置自动获取上两级目录作为根目录
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # 兼容 Jupyter 交互式窗口等环境
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'model' else os.getcwd()

# 解决画图中文乱码及负号显示问题
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun', 'Songti SC', 'STSong']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 检查计算设备 (如果有GPU则自动使用CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 读入数据与预处理
print("="*80)
file_path = os.path.join(BASE_DIR, 'data', 'MLP', '数据集2.xlsx')
print(f"正在读取数据文件: {file_path}")

df = pd.read_excel(file_path)

# 清理列名的首尾空格
df.columns = df.columns.astype(str).str.strip()

if '序号' in df.columns:
    df = df.drop(columns=['序号'])

target_col = 'CB转化率'
X_raw = df.drop(columns=[target_col])
y_raw = df[target_col].values.reshape(-1, 1)
feature_names = X_raw.columns.tolist()

# 2. 划分训练集和测试集 (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw.values, y_raw, test_size=0.2, random_state=42
)

# 3. 提取供全局重训和最终评估的数据
# 注意：为避免交叉验证数据泄露，CV内部将会独立做标准化
# 这里抽取一个全局 Scaler，专用于最后的全局(Test/Train_final)步骤
scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train)
X_test_scaled = scaler_final.transform(X_test)

# 4. 构建PyTorch MLP模型结构
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout_rate=0.0):
        super(MLP, self).__init__()
        
        layers = []
        curr_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.BatchNorm1d(h))  # 加入Batch Normalization以加速收敛
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            curr_dim = h
        # 输出层，回归任务输出维度为 1
        layers.append(nn.Linear(curr_dim, 1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# 训练和验证单次折叠(Fold)的函数
def train_and_eval_fold(X_tr, y_tr, X_val, y_val, params):
    # 将数据转换为张量并分配到设备
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    dataset_tr = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    loader_tr = DataLoader(dataset_tr, batch_size=params['batch_size'], shuffle=True, drop_last=False)
    
    model = MLP(X_tr.shape[1], params['hidden_layers'], params.get('dropout', 0.0)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params.get('weight_decay', 0))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience = params.get('patience', 40)  # 容忍epoch数早停机制
    patience_counter = 0
    
    for epoch in range(params['epochs']):
        model.train()
        for batch_x, batch_y in loader_tr:
            if batch_x.size(0) < 2:
                continue  # 保护 BN 层，跳过单样本批次
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            
        scheduler.step(val_loss)
        
        # 早停判断
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1  # 记录最佳 epoch (从 1 算起)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    # 加载最佳验证集结果权重
    model.load_state_dict(best_model_state)
        
    model.eval()
    with torch.no_grad():
        X_tr_t = torch.FloatTensor(X_tr).to(device)
        tr_pred = model(X_tr_t).cpu().numpy()
        val_pred = model(X_val_t).cpu().numpy()
        
    t_rmse = np.sqrt(mean_squared_error(y_tr, tr_pred))
    v_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    t_r2 = r2_score(y_tr, tr_pred)
    v_r2 = r2_score(y_val, val_pred)
    
    return t_rmse, v_rmse, t_r2, v_r2, model, best_epoch

# 专门用于最终全量数据固轮次重训的函数
def train_full_model(X_tr, y_tr, X_te, y_te, params):
    dataset_tr = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    loader_tr = DataLoader(dataset_tr, batch_size=params['batch_size'], shuffle=True, drop_last=False)
    
    model = MLP(X_tr.shape[1], params['hidden_layers'], params.get('dropout', 0.0)).to(device)
    criterion = nn.MSELoss()
    # 纯固定轮次，不使用学习率衰减和早停
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params.get('weight_decay', 0))
    
    X_te_t = torch.FloatTensor(X_te).to(device)
    y_te_t = torch.FloatTensor(y_te).to(device)
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    
    train_losses, test_losses = [], []
    train_r2s, test_r2s = [], []
    
    for epoch in range(params['epochs']):
        model.train()
        for batch_x, batch_y in loader_tr:
            if batch_x.size(0) < 2:
                continue  # 保护 BN 层，跳过单样本批次
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
        # 记录每个 epoch 后的 Loss 和 R2
        model.eval()
        with torch.no_grad():
            tr_pred = model(X_tr_t)
            te_pred = model(X_te_t)
            
            y_tr_t = torch.FloatTensor(y_tr).to(device)
            tr_loss = criterion(tr_pred, y_tr_t).item()
            te_loss = criterion(te_pred, y_te_t).item()
            
            tr_r2 = r2_score(y_tr, tr_pred.cpu().numpy())
            te_r2 = r2_score(y_te, te_pred.cpu().numpy())
            
            # 使用 np.sqrt 记录 RMSE 进行绘图
            train_losses.append(np.sqrt(tr_loss))
            test_losses.append(np.sqrt(te_loss))
            train_r2s.append(tr_r2)
            test_r2s.append(te_r2)
            
    return model, train_losses, test_losses, train_r2s, test_r2s

# 5. 定义超参数网格 (根据MLP容量设置)
param_grid = {
    # 'hidden_layers': [[64, 32, 16], [32, 16,8]],
    # 'lr': [0.001, 0.002,0.003],
    # 'batch_size': [32,64],
    # 'epochs': [500],                     # 依靠早停机制停止
    # 'weight_decay': [0.02],       # L2正则化防过拟合
    # 'dropout': [0.2],
    # 'patience': [40]  # 容忍epoch数早停机制
    'hidden_layers': [[64,32,16]],  # 两种容量的三层隐藏层结构，逐渐减小以形成金字塔结构
    'lr': [0.001],  # 更细粒度的学习率搜索，适应不同容量网络的收敛特性
    'batch_size': [32],
    'epochs': [800],
    'weight_decay': [0.0008],
    'dropout': [0],
    'patience': [60],
}
grid = list(ParameterGrid(param_grid))

print(f"数据处理完毕。训练集规模：{X_train_scaled.shape} | 测试集规模：{X_test_scaled.shape}")
print("-" * 80)
print(f"开始执行 MLP 网格搜索与 5 折交叉验证...")
print(f"当前计算设备: {device.type.upper()}")
print(f"搜索空间包含 {len(grid)} 种组合。\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
best_val_rmse = float('inf')
best_params = None
best_fold_details = None

# 6. 带进度条的网格搜索
for params in tqdm(grid, desc="MLP-5折交叉验证进度"):
    f_tr_rmse, f_val_rmse, f_tr_r2, f_val_r2, f_epochs = [], [], [], [], []
    
    for train_idx, val_idx in kf.split(X_train):
        X_kf_train_raw, X_kf_val_raw = X_train[train_idx], X_train[val_idx]
        y_kf_train, y_kf_val = y_train[train_idx], y_train[val_idx]
        
        # 折内独立标准化以防数据泄露
        kf_scaler = StandardScaler()
        X_kf_train = kf_scaler.fit_transform(X_kf_train_raw)
        X_kf_val = kf_scaler.transform(X_kf_val_raw)
        
        t_rmse, v_rmse, t_r2, v_r2, _, early_epoch = train_and_eval_fold(
            X_kf_train, y_kf_train, X_kf_val, y_kf_val, params
        )
        f_tr_rmse.append(t_rmse)
        f_val_rmse.append(v_rmse)
        f_tr_r2.append(t_r2)
        f_val_r2.append(v_r2)
        f_epochs.append(early_epoch)
        
    mean_v_rmse = np.mean(f_val_rmse)
    
    # 存入结果
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
            'tr_r2': f_tr_r2, 'val_r2': f_val_r2,
            'epochs': f_epochs
        }

# 7. 全网打印与展示
print("\n" + "="*80)
print(f"网格搜索找到的最佳参数组合:")
for k, v in best_params.items():
    print(f"    {k:<13}: {v}")
print("="*80)

# 把5折结果放在一起方便看
print("\n--- 最佳参数下的 5 折交叉验证详细过程 ---")
print(f"{'Fold':<8} | {'Train RMSE (Loss)':<17} | {'Val RMSE (Loss)':<17} | {'Train R2':<10} | {'Val R2':<10}")
print("-" * 80)
for i in range(5):
    print(f"Fold {i+1:<3} | {best_fold_details['tr_rmse'][i]:<17.4f} | {best_fold_details['val_rmse'][i]:<17.4f} | {best_fold_details['tr_r2'][i]:<10.4f} | {best_fold_details['val_r2'][i]:<10.4f}")
print("-" * 80)
print(f"{'Mean':<8} | {np.mean(best_fold_details['tr_rmse']):<17.4f} | {np.mean(best_fold_details['val_rmse']):<17.4f} | {np.mean(best_fold_details['tr_r2']):<10.4f} | {np.mean(best_fold_details['val_r2']):<10.4f}")
print("="*80)

# 汇总不同参数的损失情况表
print("\n--- 全网格搜索不同参数下的汇总组合 (按 Val RMSE 升序排序) ---")
print(f"{'hidden_layers':<15} | {'lr':<6} | {'batch':<5} | {'wd':<5} | {'drp':<3} | {'Train RMSE':<10} | {'Val RMSE':<10} | {'Train R2':<8} | {'Val R2':<8}")
print("-" * 105)
results_sorted = sorted(results, key=lambda x: x['val_rmse'])
for res in results_sorted:
    p_hl = str(res['params']['hidden_layers'])
    p_lr = str(res['params']['lr'])
    p_bs = str(res['params']['batch_size'])
    p_wd = str(res['params']['weight_decay'])
    p_dp = str(res['params']['dropout'])
    best_flag = "(*Opt)" if res['params'] == best_params else ""
    print(f"{p_hl:<15} | {p_lr:<6} | {p_bs:<5} | {p_wd:<5} | {p_dp:<3} | {res['train_rmse']:<10.4f} | {res['val_rmse']:<10.4f} | {res['train_r2']:<8.4f} | {res['val_r2']:<8.4f} {best_flag}")

# 8. 在独立测试集(20%)验证之前，使用全量历史训练数据重训最终模型
# 为了利用 100% (80%总体) 的历史数据，不再抽取验证集。
# 先获取最优参数在 CV 时平均的早停 epoch
# CV 仅使用 80% 训练集，全量(100%)重训时为了充分收敛，适当放大轮次 (x1.2)
# 并加一个上限保护，不超出网格预设的设计极大值
avg_early_stop_epoch = min(
    int(np.mean(best_fold_details['epochs']) * 1.2),
    param_grid['epochs'][0]
)
print(f"提取出 CV 平均早停轮次 (放大1.2倍且受限): {avg_early_stop_epoch} epochs，开始全量固定轮次重训...")
best_params_fixed = best_params.copy()
best_params_fixed['epochs'] = avg_early_stop_epoch

# 直接根据提取的固定的轮次重训，不传假验证集，不再触发带早停检测的逻辑
best_model, tr_losses, te_losses, tr_r2s, te_r2s = train_full_model(X_train_scaled, y_train, X_test_scaled, y_test, best_params_fixed)

try:
    plot_dir = os.path.join(BASE_DIR, 'data', 'MLP', 'Model_plots_MLP')
    os.makedirs(plot_dir, exist_ok=True)
except NameError:
    plot_dir = os.path.join(os.getcwd(), 'Model_plots_MLP')
    os.makedirs(plot_dir, exist_ok=True)

best_model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    test_pred = best_model(X_test_t).cpu().numpy()
    train_pred = best_model(X_train_t).cpu().numpy()

train_rmse_final = np.sqrt(mean_squared_error(y_train, train_pred))
train_r2_final = r2_score(y_train, train_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

# 1. 学习曲线 (Training vs Testing Loss/R2)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

epochs_range = list(range(1, len(tr_losses)+1))
axes[0].plot(epochs_range, tr_losses, color='#2B4A9A', linewidth=2, label='Training Loss')
axes[0].plot(epochs_range, te_losses, color='#DE4242', linewidth=2, label='Test Loss')
axes[0].set_xlabel('Epochs', fontsize=18, fontweight='bold')
axes[0].set_ylabel('Loss (RMSE)', fontsize=18, fontweight='bold')
axes[0].set_title('Training and Testing Loss Comparison', fontsize=20, fontweight='bold', pad=15)
axes[0].tick_params(axis='both', labelsize=14)
axes[0].set_xlim(left=0, right=max(epochs_range)+2)
axes[0].grid(True, linestyle='--', alpha=0.4)
axes[0].legend(fontsize=20, loc='upper right', prop={'family': 'Times New Roman', 'size': 20})
for spine in axes[0].spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

axes[1].plot(epochs_range, tr_r2s, color='#2B4A9A', linewidth=2, label='Training R²')
axes[1].plot(epochs_range, te_r2s, color='#DE4242', linewidth=2, label='Test R²')
axes[1].set_xlabel('Epochs', fontsize=18, fontweight='bold')
axes[1].set_ylabel('R² Score', fontsize=18, fontweight='bold')
axes[1].set_title('Training and Testing R² Comparison', fontsize=20, fontweight='bold', pad=15)
axes[1].tick_params(axis='both', labelsize=14)
axes[1].set_xlim(left=0, right=max(epochs_range)+2)
axes[1].grid(True, linestyle='--', alpha=0.4)
axes[1].legend(fontsize=20, loc='lower right', prop={'family': 'Times New Roman', 'size': 20})
for spine in axes[1].spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "mlp_learning_curves.png"), dpi=300, bbox_inches='tight')
plt.close()

# 提取5折数据
cv_rmses = best_fold_details['val_rmse']
cv_r2s = best_fold_details['val_r2']
avg_cv_rmse = np.mean(cv_rmses)
avg_cv_r2 = np.mean(cv_r2s)

labels = ['Train'] + [f'Fold {i+1}' for i in range(5)] + ['CV Avg', 'Test']
rmse_vals = [train_rmse_final] + list(cv_rmses) + [avg_cv_rmse, test_rmse]
r2_vals = [train_r2_final] + list(cv_r2s) + [avg_cv_r2, test_r2]

# 2. 8-Bar 性能柱状图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
bar_width = 0.5
colors_rmse = ['#5D98C1'] + ['#F4D03F']*5 + ['#D4AF37', '#E9826F']
colors_r2 = ['#5D98C1'] + ['#F4D03F']*5 + ['#D4AF37', '#E9826F']
# colors_r2 = ['#5D98C1'] + ['#A8C6DA']*5 + ['#871719', '#E9826F']

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
    ax1.text(rect.get_x() + rect.get_width()/2., h + max(rmse_vals)*0.015, f'{h:.4f}', ha='center', va='bottom', fontsize=16, fontfamily='Times New Roman', fontweight='bold')
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
    ax2.text(rect.get_x() + rect.get_width()/2., h + max_r2_val*0.015, f'{h:.4f}', ha='center', va='bottom', fontsize=16, fontfamily='Times New Roman', fontweight='bold')
for spine in ax2.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "mlp_performance_bars.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3. 散点图
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
plt.title('MLP', fontsize=20, fontweight='bold')
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

plt.text(0.95, 0.05, f'Train R²: {train_r2_final:.4f}', transform=ax.transAxes, fontsize=30, color='#5D98C1', ha='right', fontfamily='Times New Roman')
plt.text(0.95, 0.12, f'Test R²: {test_r2:.4f}', transform=ax.transAxes, fontsize=30, color='#E9826F', ha='right', fontfamily='Times New Roman')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "mlp_scatter_plot.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"评估与图表已保存至: {plot_dir}")

print("\n" + "="*80)
print("--- 独立测试集 (20%未见数据) 最终性能评估 ---")
print(f"Test RMSE (Loss) = {test_rmse:.4f}")
print(f"Test R2          = {test_r2:.4f}")
print("="*80)

# 9. SHAP 特征重要性分析
print("\n--- 正在进行 SHAP 特征重要性分析 (使用模型完全解耦的黑盒 KernelExplainer)... ---")
best_model.eval()
def predict_fn(X_np):
    X_t = torch.FloatTensor(X_np).to(device)
    with torch.no_grad():
        # 展平为 1D 数组，确保输出 (samples,) 匹配 KernelExplainer 所需的单输出规范
        return best_model(X_t).cpu().numpy().flatten()

# 使用 shap.kmeans 精简代表性背景数据（根据数据量动态调整，最多 50）
background = shap.kmeans(X_train_scaled, min(50, len(X_train_scaled)))
explainer = shap.KernelExplainer(predict_fn, background)

# 在最终的全量测试集上计算SHAP值
X_test_explainer = X_test_scaled
X_vis = pd.DataFrame(X_test, columns=feature_names)

# KernelExplainer 会显示进度条，屏蔽不必要的警告
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    shap_values = explainer.shap_values(X_test_explainer, nsamples=100)

shap_values_arr = np.array(shap_values)

# 处理多余维度 (由于 predict_fn 已展平，此时直接是 2D 数组)
if len(shap_values_arr.shape) == 3:
    if shap_values_arr.shape[0] == 1:
        shap_values_arr = shap_values_arr[0]
    elif shap_values_arr.shape[2] == 1:
        shap_values_arr = shap_values_arr[:, :, 0]

output_dir = os.path.join(BASE_DIR, 'data', 'MLP', 'SHAP_plots_MLP')
os.makedirs(output_dir, exist_ok=True)

# (0) 绘制自定义 SHAP 柱状图 (深红色, 横向)
plt.figure(figsize=(10, 8))
# 为了与 shap summary_plot 保持类似的外观，我们自定义绘制
mean_abs_impact = np.abs(shap_values_arr).mean(axis=0)
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
bar_path = os.path.join(output_dir, "mlp_shap_bar_plot.png")
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSHAP 贡献率柱状图 已保存至:\n  -> {bar_path}")

# (1) 绘制全局 Summary Plot (摘要蜂群图)
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

summary_path = os.path.join(output_dir, "mlp_shap_summary_plot.png")
plt.tight_layout()
plt.savefig(summary_path, dpi=300)
plt.close()
print(f"SHAP 摘要蜂群图 (Summary Plot) 已保存至:\n  -> {summary_path}")

# (2) 依赖图 Dependence Plot (相关性图)
mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
sorted_indices = np.argsort(mean_abs_shap)[::-1]
top_feature = feature_names[int(sorted_indices[0])]

plt.figure(figsize=(8, 6))
shap.dependence_plot(top_feature, shap_values_arr, X_vis, show=False)

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

safe_feature_name = top_feature.replace('/', '_').replace(':', '')
dep_path = os.path.join(output_dir, f"mlp_shap_dependence_plot_{safe_feature_name}.png")
plt.tight_layout()
plt.savefig(dep_path, dpi=300)
plt.close()
print(f"SHAP 依赖图 (Dependence Plot) [基于最重要的特征:{top_feature}] 已保存至:\n  -> {dep_path}")

# (3) 打印所有特征的贡献度排序
print("\n--- 各特征的平均绝对 SHAP 贡献度 (重要性按降序排列) ---")
for idx in sorted_indices:
    print(f"{feature_names[idx]:<20}: {mean_abs_shap[idx]:.5f}")

print("="*80)
print("MLP 模型所有流程执行完毕！")