"""
Created on 8 May 2024 12:23 PM
@author: nabin

Usage:
- generate pdb with color spectrum
"""

# 寫入pdb的B-factor
# 輸出殘基信心分數的散點圖

from Bio import PDB
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import pearsonr

############### new ###############
def save_scores_to_pdb(conf_score_file, input_pdb_file, output_pdb_file):
    # 讀取信心分數 CSV 檔案
    conf_df = pd.read_csv(conf_score_file)
    # 提取預測的氨基酸類型信心分數列表
    scores = conf_df['Pred AA Prob'].to_list()

    # 解析 PDB 結構
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', input_pdb_file)

    # 自動計算殘基號起點 (offset)
    all_res_nums = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag, resseq, icode = residue.id
                if hetflag == ' ':
                    all_res_nums.append(resseq)
    if not all_res_nums:
        raise ValueError("PDB 裡找不到任何標準殘基")
    start_res = min(all_res_nums)

    # 只為 CA 原子設定 B-factor
    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag, resseq, icode = residue.id
                if hetflag != ' ':
                    continue
                idx = resseq - start_res  # 轉為 0-index
                if 0 <= idx < len(scores):
                    score = scores[idx]
                else:
                    score = 0.0  # 超出範圍時預設 0.0

                if 'CA' in residue:  # 僅改 CA 原子
                    residue['CA'].set_bfactor(score)

    # 輸出新的 PDB
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_file)

    # 刪除原始 PDB (可選)
    if os.path.exists(input_pdb_file):
        os.remove(input_pdb_file)
        
# ---------------------- 原始 ----------------------
# # 將預測分數存入 PDB 檔案，並使用 B-factor 來表示殘基信心分數
# def save_scores_to_pdb(conf_score_file, input_pdb_file, output_pdb_file):
#     # 讀取信心分數 CSV 檔案
#     conf_df = pd.read_csv(conf_score_file)
#     # 提取預測的氨基酸類型信心分數
#     scores = conf_df['Pred AA Prob'].to_list()

#     # 解析 PDB 結構
#     parser = PDB.PDBParser(QUIET=True)
#     structure = parser.get_structure('protein', input_pdb_file)

#     # Iterate through atoms and assign scores as B-factors (遍歷蛋白質結構中的所有原子，將預測分數存入 B-factor)
#     for atom in structure.get_atoms():
#         residue_id = atom.get_parent().get_id()[1] # 獲取殘基 ID
#         score = scores[residue_id] # 取得該殘基對應的信心分數
#         atom.set_bfactor(score) # 設置 B-factor 為信心分數

#     # Write modified structure to output PDB file (將修改後的 PDB 結構儲存到輸出 PDB 檔案)
#     io = PDB.PDBIO()
#     io.set_structure(structure)
#     io.save(output_pdb_file)
    
#     # 刪除原始 PDB 檔案（可選）
#     if os.path.exists(input_pdb_file):
#         os.remove(input_pdb_file)


# 生成殘基信心分數的散點圖，並計算相關係數
def generate_plot(conf_score_file, plot_filename):
    shapes = ['o', 's', '^', 'x', 'v', 'D', 'p', '*', '>', '<', 'h', '+', '|', '.', '1', '2', '3', '4', '8', 'd']
    df = pd.read_csv(conf_score_file) # 讀取信心分數 CSV 檔案
    residue = df['Residue'].to_list() # 提取殘基名稱
    ca_confidence = df['Pred CA Prob'].to_list() # 提取 CA 信心分數
    aa_confidence = df['Pred AA Prob'].to_list() # 提取氨基酸類型信心分數
    unique_residues = list(set(residue)) # 找出所有獨特的殘基類型
    num_unique_residues = len(unique_residues)

    # 計算 CA 信心分數與 AA 信心分數的皮爾遜相關係數
    correlation_coeff, p_value = pearsonr(ca_confidence, aa_confidence)
    avg_ca_conf = sum(ca_confidence)/len(ca_confidence) # CA 信心分數的平均值
    avg_aa_conf = sum(aa_confidence)/len(aa_confidence) # AA 信心分數的平均值
    
    # Plotting (開始繪製散點圖)
    plt.figure(figsize=(10, 6)) # 設定圖像大小
    for i, res in enumerate(unique_residues):
        shape = shapes[i % num_unique_residues]  # Cycle through shapes  (循環使用不同的標記形狀)
        indices = [idx for idx, r in enumerate(residue) if r == res] # 找出該殘基對應的索引
        plt.scatter(
            [ca_confidence[idx] for idx in indices], # CA 信心分數作為 x 軸
            [aa_confidence[idx] for idx in indices], # AA 信心分數作為 y 軸
            label=res, # 標記殘基名稱
            marker=shape, # 設定不同的標記形狀
            edgecolors='black', # 設定邊框顏色為黑色
            s=100) # 設定標記大小

    # 設定圖表標題與軸標籤
    plt.title('Residue Scores')
    plt.xlabel('CA Confidence')
    plt.ylabel('Amino Acid Type Confidence')
    plt.legend(title='Residue') # 顯示圖例
    plt.grid(False)  # Turn off grid # 關閉網格線
    # plt.xlim(0.4, 0.7)  # Set x-axis limit
    # plt.ylim(0, 1)  # Set y-axis limit
    
    # 調整佈局，確保標籤不會被切掉
    plt.tight_layout()
    # plt.show()
    
    # 儲存圖表為高解析度圖片
    plt.savefig(plot_filename, bbox_inches='tight', dpi=1000)
    
    # 返回 CA 和 AA 信心分數的平均值
    return avg_ca_conf, avg_aa_conf