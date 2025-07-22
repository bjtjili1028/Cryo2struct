
"""
Created on 8 May 2024 11:16 AM
@author: nabin

Usage:
- generates confidence scores
"""

# 輸出沒有b-factor的pdb
# 輸出預測值的csv

import pickle
import pandas as pd
import os
import csv

# 單字母氨基酸代碼對應到三字母氨基酸代碼的映射表
restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}

# 氨基酸的 One-Hot 編碼表示
res_one_hot = {
    'ALA' : "(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
    'ARG' : "(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
    'ASN' : "(0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
    'ASP' : "(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
    'CYS' : "(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
    'GLN' : "(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
    'GLU' : "(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
    'GLY' : "(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
    'HIS' : "(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
    'ILE' : "(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
    'LEU' : "(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
    'LYS' : "(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)",
    'MET' : "(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)",
    'PHE' : "(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)",
    'PRO' : "(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)",
    'SER' : "(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)",
    'THR' : "(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)",
    'TRP' : "(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)",
    'TYR' : "(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0)",
    'VAL' : "(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)",
}

# 轉換 One-Hot 編碼為 Python 元組格式
res_one_hot1 = {key: tuple(map(int, value.strip('()').split(','))) for key, value in res_one_hot.items()}

# 產生機率分數檔案
# 這個函式會將殘基名稱、CA機率、AA類型機率寫入 CSV 檔案

def res_prob_score_files(save_prob_score_file, seq_list, seq_list_conf, ca_list, ami_list):
    csv_headers = ["Residue", "CA Prob", "AA Type Prob Emi"] # CSV 標題
    with open(save_prob_score_file,"w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_headers)
        for a in range(len(seq_list)):
            residue = restype_1to3[seq_list_conf[a]] # 轉換殘基代碼為三字母格式
            ca_prob = round(ca_list[a], 3) # 四捨五入 CA 機率
            aa_emi = round(ami_list[a], 3) # 四捨五入 AA 機率
            write_c = [residue, ca_prob, aa_emi]
            csv_writer.writerow(write_c)  # 寫入 CSV

# 預測 CA 機率
def conf_scores_ca(model, test_data_df):
    # 讀取訓練好的模型
    with open(model, 'rb') as f:
        model = pickle.load(f)
    
    # 提取特徵
    X_test = test_data_df[['CA Prob']]  # Features
    
    # 進行預測
    test_preds_probs = model.predict_proba(X_test)
    pred_probs = [x[1] for x in test_preds_probs]
    
    # 加入預測機率
    test_data_df['Pred CA Prob'] = pred_probs
    return test_data_df

# 預測 AA 機率
def conf_scores_aa(model, test_data):
    with open(model, 'rb') as f:
        model = pickle.load(f)

    df_test = pd.read_csv(test_data)
    df_test1 = df_test.copy()
    # 轉換 One-Hot
    df_test['Residue One Hot'] = df_test['Residue'].map(res_one_hot1)
    
    one_hot_df = pd.DataFrame(df_test['Residue One Hot'].tolist(), columns=['Feature_{}'.format(i) for i in range(len(df_test['Residue One Hot'].iloc[0]))])
    df_test = pd.concat([df_test, one_hot_df], axis=1)
    X_test = df_test[['CA Prob', 'AA Type Prob Emi', 'Feature_0', 'Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8',
                        'Feature_9', 'Feature_10', 'Feature_11', 'Feature_12', 'Feature_13', 'Feature_14', 'Feature_15', 'Feature_16', 'Feature_17', 'Feature_18',
                    'Feature_19']]  # Features # 20種氨基酸類型

    test_preds_probs = model.predict_proba(X_test)
    pred_probs = [x[1] for x in test_preds_probs]
    df_test1['Pred AA Prob'] = pred_probs
    return df_test1

# 生成最終的信心分數並儲存到 CSV
def gen_conf_scores(prob_scores, save_path, trained_regression_model_aa, trained_regression_model_ca ):
    
    # 將預測出來的CA和AA加入原始CSV中(res_prob_score_files) 
    test_data_df_aa = conf_scores_aa(model=trained_regression_model_aa, test_data=prob_scores) 
    test_data_df_ca = conf_scores_ca(model=trained_regression_model_ca, test_data_df=test_data_df_aa)
    
    # 刪除原始CA和AA
    test_data_df_ca = test_data_df_ca.drop(columns='CA Prob') 
    test_data_df_ca = test_data_df_ca.drop(columns='AA Type Prob Emi')
    
    # 若檔案已存在，則刪除
    if os.path.exists(save_path):
        os.remove(save_path)

    # 若輸入的機率分數檔案存在，也刪除
    if os.path.exists(prob_scores):
        os.remove(prob_scores)
    
    # 將最終結果存為 CSV
    test_data_df_ca.to_csv(save_path, index=False)