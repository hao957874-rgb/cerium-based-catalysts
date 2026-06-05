import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap
import matplotlib.pyplot as plt
import seaborn as sns
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
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
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
            
            train_losses.append(tr_loss)
            test_losses.append(te_loss)
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

# 绘制与保存 Loss 和 R2 曲线图
try:
    output_dir_plots = os.path.join(BASE_DIR, 'data', 'MLP', 'SHAP_plots_MLP')
    os.makedirs(output_dir_plots, exist_ok=True)
except NameError:
    output_dir_plots = os.path.join(os.getcwd(), 'SHAP_plots_MLP')
    os.makedirs(output_dir_plots, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(tr_losses, label='Train Loss (MSE)', color='blue')
plt.plot(te_losses, label='Test Loss (MSE)', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Testing Loss over Epochs')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir_plots, "mlp_loss_curve.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(tr_r2s, label='Train R2', color='blue')
plt.plot(te_r2s, label='Test R2', color='red')
plt.xlabel('Epoch')
plt.ylabel('R2 Score')
plt.title('Training and Testing R2 over Epochs')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir_plots, "mlp_r2_curve.png"), dpi=300)
plt.close()
print(f"Loss 和 R2 演变曲线已保存至:\n  -> {output_dir_plots}")

best_model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    test_pred = best_model(X_test_t).cpu().numpy()
    
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

print("\n" + "="*80)
print("--- 独立测试集 (20%未见数据) 最终性能评估 ---")
print(f"Test RMSE (Loss) = {test_rmse:.4f}")
print(f"Test R2          = {test_r2:.4f}")
print("="*80)

# ================= 新增：多 seed 稳定性验证模块 =================
print("\n" + "="*80)
print("--- 准备进行多 Seed 稳定性验证 ---")
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42, 20, 30, 50, 123, 256, 512, 1024, 2048, 4096]
seed_metrics = []

best_global_r2 = -float('inf')
best_global_model = None
best_global_scaler = None
best_global_X_test = None
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
        X_raw.values, y_raw, test_size=0.2, random_state=seed
    )
    
    # 3. 全局scaler
    scaler_s = StandardScaler()
    X_train_scaled_s = scaler_s.fit_transform(X_train_s)
    X_test_scaled_s = scaler_s.transform(X_test_s)
    
    kf_s = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_val_rmse_s = float('inf')
    best_params_s = None
    best_fold_epochs_s = None
    
    # 4. 网格搜索验证 (直接复用原定义的 param_grid)
    for params in grid:
        f_val_rmse_s = []
        f_epochs_s = []
        for train_idx, val_idx in kf_s.split(X_train_s):
            X_kf_train_raw, X_kf_val_raw = X_train_s[train_idx], X_train_s[val_idx]
            y_kf_train, y_kf_val = y_train_s[train_idx], y_train_s[val_idx]
            
            # 折内独立标准化
            kf_scaler_s = StandardScaler()
            X_kf_train = kf_scaler_s.fit_transform(X_kf_train_raw)
            X_kf_val = kf_scaler_s.transform(X_kf_val_raw)
            
            # 使用现有的 train_and_eval_fold
            _, v_rmse, _, _, _, early_epoch = train_and_eval_fold(
                X_kf_train, y_kf_train, X_kf_val, y_kf_val, params
            )
            f_val_rmse_s.append(v_rmse)
            f_epochs_s.append(early_epoch)
            
        mean_v_rmse_s = np.mean(f_val_rmse_s)
        if mean_v_rmse_s < best_val_rmse_s:
            best_val_rmse_s = mean_v_rmse_s
            best_params_s = params
            best_fold_epochs_s = f_epochs_s
            
    # 5. 固轮次重训
    avg_early_stop_epoch_s = min(
        int(np.mean(best_fold_epochs_s) * 1.2),
        param_grid['epochs'][0]
    )
    best_params_fixed_s = best_params_s.copy()
    best_params_fixed_s['epochs'] = avg_early_stop_epoch_s
    
    # 使用现有的 train_full_model
    best_model_s, _, _, _, _ = train_full_model(
        X_train_scaled_s, y_train_s, X_test_scaled_s, y_test_s, best_params_fixed_s
    )
    
    # 6. 独立测试集评估
    best_model_s.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test_scaled_s).to(device)
        test_pred_s = best_model_s(X_test_t).cpu().numpy()
        
    test_rmse_s = np.sqrt(mean_squared_error(y_test_s, test_pred_s))
    test_r2_s = r2_score(y_test_s, test_pred_s)
    
    # 7. 记录结果
    seed_metrics.append({
        'seed': seed,
        'test_rmse': test_rmse_s,
        'test_r2': test_r2_s
    })
    
    # 8. 更新全局最佳模型信息 (用于备存，不干扰 SHAP)
    if test_r2_s > best_global_r2:
        best_global_r2 = test_r2_s
        best_global_model = best_model_s
        best_global_scaler = scaler_s
        best_global_X_test = X_test_s
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

# 可视化模块
print("\n--- 正在生成多 Seed 稳定性验证可视化图表 (四张独立的图) ---")
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

multi_seed_output_dir = os.path.join(BASE_DIR, 'data', 'MLP', 'SHAP_plots_MLP_seed')
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
fig1 = plt.figure(figsize=(8, 6), dpi=300)
ax1 = plt.gca()
bars1 = ax1.bar(seed_strs, r2_arr, color='#5B9BD5', edgecolor='black', width=0.6, zorder=3)
ax1.axhline(mean_r2, color='red', linestyle='--', linewidth=2, zorder=4, label=f'Mean R²: {mean_r2:.4f}')
ax1.legend(prop={'family': 'Times New Roman', 'size': 18})
ax1.set_title('Test R² 随随机种子验证表现', fontdict=title_font_zh)
ax1.set_xlabel('Random Seed', fontdict=label_font)
ax1.set_ylabel('Test R²', fontdict=label_font)
ax1.set_xlim(-1.0, len(seed_strs)-0.2)
ax1.tick_params(axis='x', rotation=45, labelsize=14)
for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
    tick.set_fontname(tick_font)
    tick.set_fontsize(14)
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
plt.savefig(os.path.join(multi_seed_output_dir, "mlp_multi_seed_r2_bar.png"), dpi=300, bbox_inches='tight')
plt.close(fig1)

# 图2
fig2 = plt.figure(figsize=(8, 6), dpi=300)
ax2 = plt.gca()
bars2 = ax2.bar(seed_strs, rmse_arr, color='#F4D03F', edgecolor='black', width=0.6, zorder=3)
ax2.axhline(mean_rmse, color='red', linestyle='--', linewidth=2, zorder=4, label=f'Mean RMSE: {mean_rmse:.4f}')
ax2.legend(prop={'family': 'Times New Roman', 'size': 18})
ax2.set_title('Test RMSE 随随机种子验证表现', fontdict=title_font_zh)
ax2.set_xlabel('Random Seed', fontdict=label_font)
ax2.set_ylabel('Test RMSE', fontdict=label_font)
ax2.set_xlim(-1.0, len(seed_strs)-0.2)
ax2.tick_params(axis='x', rotation=45, labelsize=14)
for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
    tick.set_fontname(tick_font)
    tick.set_fontsize(14)
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
plt.savefig(os.path.join(multi_seed_output_dir, "mlp_multi_seed_rmse_bar.png"), dpi=300, bbox_inches='tight')
plt.close(fig2)

# 图3
fig3 = plt.figure(figsize=(8, 6), dpi=300)
ax3 = plt.gca()
sns.boxplot(y=r2_arr, ax=ax3, color='#D3D3D3', width=0.3, zorder=2)
sns.stripplot(y=r2_arr, ax=ax3, color='#E77C6E', size=6, jitter=True, zorder=3)
ax3.set_title('Distribution of Test R²', fontdict=title_font)
ax3.set_xlabel('MLP Model', fontdict=label_font)
ax3.set_ylabel('Test R²', fontdict=label_font)
ax3.set_xticks([0])
ax3.set_xticklabels([''])
for tick in ax3.get_xticklabels() + ax3.get_yticklabels():
    tick.set_fontname(tick_font)
    tick.set_fontsize(14)
ax3.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(multi_seed_output_dir, "mlp_multi_seed_r2_box.png"), dpi=300, bbox_inches='tight')
plt.close(fig3)

# 图4
fig4 = plt.figure(figsize=(8, 6), dpi=300)
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
plt.savefig(os.path.join(multi_seed_output_dir, "mlp_multi_seed_scatter.png"), dpi=300, bbox_inches='tight')
plt.close(fig4)

print(f"多 Seed 稳定性验证四张独立图表已保存至:\n  -> {multi_seed_output_dir}")
# ================================================================

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

# (1) 绘制全局 Summary Plot (摘要蜂群图)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_arr, X_vis, show=False)
summary_path = os.path.join(output_dir, "mlp_shap_summary_plot.png")
plt.tight_layout()
plt.savefig(summary_path, dpi=300)
plt.close()
print(f"\nSHAP 摘要蜂群图 (Summary Plot) 已保存至:\n  -> {summary_path}")

# (2) 依赖图 Dependence Plot (相关性图)
mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
sorted_indices = np.argsort(mean_abs_shap)[::-1]
top_feature = feature_names[int(sorted_indices[0])]

plt.figure(figsize=(8, 6))
shap.dependence_plot(top_feature, shap_values_arr, X_vis, show=False)
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