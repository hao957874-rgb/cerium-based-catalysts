import pandas as pd
import numpy as np
import os
import glob
import shap
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
ensemble_pairs = []

# 3. 定义网格搜索选参空间
param_grid = {
    'kernel': ['rbf'],
    'C': [10],
    'gamma': [0.12],
    'epsilon': [0.01]
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
            val_pred = np.clip(val_pred, 0, 1)  # 物理约束：转化率在0~1之间
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
    
    ensemble_pairs.append({
        'seed': seed,
        'scaler': scaler_final,
        'model': final_model
    })
    
    # 盲测评估
    test_pred = final_model.predict(X_test_scaled)
    test_pred = np.clip(test_pred, 0, 1)  # 物理约束：转化率在0~1之间
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
print("--- 10个不同 Seed 的独立盲测结果汇总 ---")
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

# ==============================================================================
# --- 新增模块：集成预测阶段 ---
# ==============================================================================
predict_dir = r'D:\vsshujubao\CB\predict'
output_dir = os.path.join(predict_dir, '预测结果')
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*80)
print(">>> 启动多 Seed 集成预测模块")
print("================================================================================")

# 重新从训练集文件读取范围，解耦依赖
df_train_ref = pd.read_excel(file_path)
if '序号' in df_train_ref.columns:
    df_train_ref = df_train_ref.drop(columns=['序号'])
train_min = df_train_ref[feature_names].min()
train_max = df_train_ref[feature_names].max()

# 直接指定需要预测的 excel 文件
target_file = os.path.join(predict_dir, '预测集_配比对照.xlsx')#路径
predict_files = [target_file] if os.path.exists(target_file) else []

if not predict_files:
    print(f"未在预测目录下找到指定文件 {target_file}，预测模块跳过。")
else:
    for file in tqdm(predict_files, desc="处理预测文件"):
        filename = os.path.basename(file)
        
        try:
            df_pred = pd.read_excel(file)
            df_pred = df_pred.reset_index(drop=True)
        except Exception as e:
            print(f"\n跳过文件 {filename}，读取失败: {e}")
            continue
            
        # 序号列处理
        index_col = None
        if '序号' in df_pred.columns:
            index_col = df_pred['序号'].copy()
            df_pred = df_pred.drop(columns=['序号'])
            
        # 检查特征列是否一致
        pred_features = df_pred.columns.tolist()
        if pred_features != feature_names:
            print(f"\n警告：跳过文件 {filename}，它的特征列与训练集不一致！")
            continue
            
        # 获取用于预测的纯特征 ndarray
        X_pred_raw = df_pred.values if hasattr(df_pred, 'values') else df_pred
        
        if X_pred_raw.shape[0] == 0:
            print(f"\n跳过文件 {filename}，该文件没有数据行。")
            continue
            
        # 用 20 个 seed 的 scaler 和 model 组合分别预测
        all_preds = []
        for pair in ensemble_pairs:
            scaler = pair['scaler']
            model = pair['model']
            
            # 使用配套 scaler 进行 transform
            X_pred_scaled = scaler.transform(X_pred_raw)
            # 预测
            pred = model.predict(X_pred_scaled)
            pred = np.clip(pred, 0, 1)  # 物理约束：转化率在0~1之间
            all_preds.append(pred)
            
        # all_preds 形状：(20, 样本数)
        all_preds = np.array(all_preds)
        
        # 聚合预测结果
        mean_preds = np.mean(all_preds, axis=0)
        std_preds = np.std(all_preds, axis=0)
        max_preds = np.max(all_preds, axis=0)
        min_preds = np.min(all_preds, axis=0)
        
        print("\n" + "="*80)
        print(f"预测文件: {filename}  共 {len(mean_preds)} 个样本")
        print("================================================================================")
        print(f"{'样本序号':<8} | {'集成预测值(均值)':<16} | {'预测不确定性(±std)':<18}")
        print("-" * 56)
        
        # 打印预览（最多打印前10行避免刷屏）
        preview_count = min(10, len(mean_preds))
        for i in range(preview_count):
            disp_idx = index_col.iloc[i] if index_col is not None else i+1
            print(f"{disp_idx:<12} | {mean_preds[i]:<21.4f} | {std_preds[i]:<18.4f}")
        if len(mean_preds) > 10:
            print("... (省略后续)")
        print("================================================================================\n")
        
        # 保存结果：保留包含序号和原特征值的数据框，并追加新增列
        result_df = df_pred[feature_names].copy()
        if index_col is not None:
            result_df.insert(0, '序号', index_col.values)
            
        # 增加预测统计列
        result_df['CB转化率_预测均值'] = mean_preds
        result_df['CB转化率_预测标准差'] = std_preds
        result_df['CB转化率_预测最大值'] = max_preds
        result_df['CB转化率_预测最小值'] = min_preds
        
        # ==================== 可信度标注评估 ====================
        out_of_bounds_counts = []
        out_of_bounds_names = []
        credibility_labels = []
        
        for i in range(len(result_df)):
            sample = result_df.iloc[i]
            
            # 1. 检查外推
            oob_feats = [
                feat for feat in feature_names
                if pd.notna(sample[feat]) and (sample[feat] < train_min[feat] or sample[feat] > train_max[feat])
            ]
            
            out_of_bounds_counts.append(len(oob_feats))
            out_of_bounds_names.append(", ".join(oob_feats) if oob_feats else "无")
            
            m_pred = mean_preds[i]
            s_pred = std_preds[i]
            
            # 收集所有成立的问题，不互相覆盖
            reasons = []
            if len(oob_feats) > 0:
                reasons.append("超出训练范围")
            if m_pred >= 1.0:
                reasons.append("预测饱和")
            if s_pred > 0.07:
                reasons.append("模型分歧大")
            if m_pred > 0.95:
                reasons.append("接近上界")

            if not reasons:
                label = "正常"
            elif any(r in reasons for r in ["超出训练范围", "预测饱和"]):
                label = "不可信 · " + " + ".join(reasons)
            else:
                label = "低可信 · " + " + ".join(reasons)
                
            credibility_labels.append(label)
            
        result_df['外推特征数'] = out_of_bounds_counts
        result_df['外推特征名'] = out_of_bounds_names
        result_df['可信度标注'] = credibility_labels
        
        # 打印分布统计
        print(f"\n--- {filename} 可信度标注统计 ---")
        label_counts = pd.Series(credibility_labels).value_counts()
        for label, count in label_counts.items():
            print(f"  {label:<20} : {count} 条")
        print("-" * 56)
        
        # 导出 Excel
        output_filepath = os.path.join(output_dir, f"预测结果_含可信度标注_预测集_配比对照.xlsx")
        result_df.to_excel(output_filepath, index=False)
        print(f"[{filename}] 预测已写入: {output_filepath}")

print("集成预测结束。")
