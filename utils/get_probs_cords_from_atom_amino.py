"""
Created on 02 March 2023 05:14:00 PM
@author: nabin

找出ca原子和胺基酸類型的共同座標

"""
import ast
import math

# 存儲碳 α (CA) 原子的座標
ca_coordinates = list()

# 存儲機率值的字典
prob_dic = dict()

#################### 利用模式切換分割方式 ##########################

def split_atom_file(probability_file_atom, split_output_ca, split_output_n, split_output_c, ca_threshold, mode=2 ):
    """
    根據 mode 選擇不同分類邏輯：
    
    mode == 1: 使用最大值分類，直接從 p1, p2, p3 中選出最大值進行分類輸出。
    mode == 2: 使用 0.4 門檻方式：
              如果 Cα (p1) >= ca_threshold 則分類為 Cα，
              否則檢查 N (p2) 與 C (p3)：
                若兩者均 < 0.4 則忽略該筆資料，
                否則選擇較大者分類為 N 或 C。
                
    每一行輸入格式假設為:
       [x, y, z], p0, p1, p2, p3
    其中:
       p0: 無原子, p1: Cα, p2: N, p3: C
    """
    with open(probability_file_atom, 'r') as infile, \
         open(split_output_ca, 'w') as f_ca, \
         open(split_output_n, 'w') as f_n, \
         open(split_output_c, 'w') as f_c:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                # 分離座標部分與機率部分，假設格式 "[x, y, z], p0, p1, p2, p3"
                coord_part, probs_part = line.split("],", 1)
                coord_str = coord_part + "]"
                coord = ast.literal_eval(coord_str)
                probs = [float(x.strip()) for x in probs_part.split(",")]
            except Exception as e:
                print(f"解析錯誤: {line}. Error: {e}")
                continue

            if len(probs) != 4:
                print(f"機率數量錯誤 (非4個機率): {line}")
                continue

            p0, p_ca, p_n, p_c = probs
            
            if mode == 1:
                # 模式 1: 最大值分類輸出（忽略門檻條件）
                # 選出最大值當作是該原子
                max_val = max(p0,p_ca, p_n, p_c)
                if max_val == p0 :
                    continue
                elif max_val == p_ca:
                    f_ca.write(f"{coord}, {p_ca}\n")
                elif max_val == p_n:
                    f_n.write(f"{coord}, {p_n}\n")
                else :
                    f_c.write(f"{coord}, {p_c}\n")
                      
                
            elif mode == 2:
                # 模式 2: 使用門檻方式分類輸出
                if p_ca >= ca_threshold :
                    f_ca.write(f"{coord}, {p_ca}\n")
                else:
                    # 當 Cα 的機率低於門檻，檢查 N 與 C 的機率
                    if p_n < ca_threshold and p_c < ca_threshold :
                        # 若 N 與 C 都低於 ca_threshold，則忽略該筆資料
                        continue
                    else:
                        # 若至少有一個超過 ca_threshold，則選擇較大者進行分類
                        if p_n >= p_c:
                            f_n.write(f"{coord}, {p_n}\n")
                        else:
                            f_c.write(f"{coord}, {p_c}\n")
                            
            else:
                print("無效的模式，請設定 mode 為 1(使用最大值分類) 或 2(使用原始方式)。")
                return

#################### 原始code #############################

def get_joint_probabity_common_threshold(probability_file_atom, probability_file_amino_atom_common, probability_file_amino, s_c, threshold, probability_file_amino_atom_common_ca_prob):
    """
    get only common carbon alphas from amino and atom files
    從氨基酸與原子機率文件中獲取共有的碳 α (CA) 原子
    """
    common_ca = dict() # 存儲共有的碳 α 原子機率值
    common_coordinate_prob = dict() # 存儲座標機率
    amino_acid_emission = dict() # 存儲氨基酸的機率發射值
    count_uncommon_atoms = 0 # 記錄不常見的原子數
    count_common_atoms = 0 # 記錄常見的原子數
    total_atom_entries = 0  # 總原子條目計數
    total_saved_ca = 0 # 記錄被儲存的碳 α 原子數

    # 讀取氨基酸機率文件，並處理其中的數據
    with open(probability_file_amino, 'r') as amino_prob:
        for line in amino_prob:
            line_a = ast.literal_eval(line) # 將字串格式的列表轉換為 Python 資料結構
            common_coordinate_prob[tuple(line_a[0])] = 1 - line_a[1] # 計算座標機率
            aa_val = list(line_a[2:]) # 取出氨基酸的發射機率值
            equal_part_add = line_a[1]  / 20 # 平均分配機率值
            aa_val = tuple([x + equal_part_add for x in aa_val]) # 更新機率值
            amino_acid_emission[tuple(line_a[0])] = aa_val # 存入氨基酸發射機率
            total_atom_entries += 1  # 增加原子計數
                 
    # 讀取原子機率文件，匹配座標並計算共同碳 α 原子的機率  
    with open(probability_file_atom, 'r') as atom_prob:
        for line in atom_prob:
            line_a = ast.literal_eval(line) # 解析行數據
            try:
                # 根據座標從 common_coordinate_prob 取得機率值並計算平方根
                # common_ca[tuple(line_a[0])] = math.sqrt(common_coordinate_prob[tuple(line_a[0])] * line_a[2])  
                
                # 原始是*line_a[2] 但因為將資料進行切分，所以修改成 line_a[1]
                common_ca[tuple(line_a[0])] = math.sqrt(common_coordinate_prob[tuple(line_a[0])] * line_a[1])
                common_ca[tuple(line_a[0])] = line_a[1] # 直接存入原子機率值
                count_common_atoms += 1  # 記錄常見的原子
    
            except KeyError:
                count_uncommon_atoms += 1 # 記錄不常見的原子

    # 開啟多個檔案來儲存結果
    save_cluster_co = open(s_c, 'a') # 儲存碳 α 原子座標
    save_cluster_prob = open(probability_file_amino_atom_common_ca_prob,'a') # 儲存共同碳 α 機率
    amino_atom_prob = open(probability_file_amino_atom_common,'a') # 儲存氨基酸與原子的機率資訊
    
    # 遍歷 common_ca 字典，篩選符合閾值條件的座標
    for k,v in common_ca.items():
        if v > threshold:
            try:
                emiss_val = amino_acid_emission[k] # 取得對應的氨基酸發射機率值
                x,y,z = k # 解析座標
                print(f"{x} {y} {z}", file=save_cluster_co) # 將座標寫入文件
                amino_atom_prob.write(f"{list(k)}") # 寫入座標數據
                save_cluster_prob.write(f"{list(k)}") # 寫入座標數據
                save_cluster_prob.write(f", {v}") # 寫入機率值
                save_cluster_prob.write(f"\n")
                for e in emiss_val:
                    amino_atom_prob.write(f", {e}") # 寫入氨基酸發射機率
                amino_atom_prob.write(f"\n")
                total_saved_ca += 1 # 記錄儲存的碳 α 原子數量
            except KeyError:
                q  = 1 # 若座標無對應的氨基酸發射機率則忽略

