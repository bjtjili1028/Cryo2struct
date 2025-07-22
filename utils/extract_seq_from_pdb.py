"""
Created on 18 April 2023 12:23 AM
@author: nabin

Usage:
- Gets sequence from pdb file along with its chain information

"""

# 引入BioPython庫中的PDB模組，用於解析PDB文件
from Bio import PDB

# 定義3字母氨基酸縮寫到1字母氨基酸的映射字典
restype_3to1 = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
    'UNK': 'U', # 未知氨基酸的處理
} 

# 定義一個字典來存儲每個鏈的氨基酸序列
chain_seq_dict = dict()


# 提取PDB文件中的氨基酸序列及其鏈信息
def extract_seq(pdb_file, atomic_chain_seq_file, atomic_seq_file):
    # 使用Bio.PDB的PDBParser來解析PDB文件
    parser = PDB.PDBParser()
    pdb_map = pdb_file # PDB文件路徑
    struct = parser.get_structure("CA", pdb_map) # 解析結構並取得模型中的CA（alpha碳）
    
    # 遍歷結構中的每個模型、鏈和殘基
    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "CA": # 只關心CA原子
                        chain_id = chain.id # 獲取鏈的ID
                        try:
                            # 將三字母氨基酸名稱轉換為一字母代碼
                            amino_name = restype_3to1[residue.resname] 
                            # 如果鏈已存在於字典中，將氨基酸名添加到該鏈的序列中
                            if chain_id in chain_seq_dict:
                                chain_seq_dict[chain_id].append(amino_name)
                            # 新鏈，創建新的列表
                            else:
                                chain_seq_dict[chain_id] = [amino_name]
                        except KeyError:
                            pass # 如果氨基酸名稱不在映射字典中，則跳過該氨基酸
    
    # 保存每條鏈的序列到atomic_chain_seq_file文件
    with open(atomic_chain_seq_file, 'w') as a_c:
        for k,v in chain_seq_dict.items():
            # 每條鏈前加上描述
            print(f">pdb2seq|Chains {k}", file=a_c)
            result = ''.join(v) # 將氨基酸序列合併成字符串
            print(result, file=a_c) # 寫入文件
    
    # 將所有鏈的序列合併為一個大序列並保存
    all_seq = list()
    with open(atomic_seq_file, 'w') as a_s:
        print(">pdb2seq|Chains A", file=a_s) # 設置序列標題
        for k,v in chain_seq_dict.items():
            result = ''.join(v) # 合併鏈的氨基酸序列
            all_seq.append(result) # 添加到總序列中
        final_result = ''.join(all_seq) # 合併所有鏈的序列
        print(final_result,file=a_s) # 將最終結果寫入文件