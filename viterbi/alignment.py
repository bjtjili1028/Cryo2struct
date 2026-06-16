""" 
Created on 25 Jan 2023 10:21 AM
Updated on 8 May 2024 3:43 PM
@author: nabin

Usage:
- Construct HMM
- Align using Viterbi

"""

import numpy as np
import scipy.stats
import os
import ctypes
import time
import glob
import re
from utils import extract_seq_from_pdb # 引入提取PDB序列的工具
from utils import bond_matching
from utils.bond_matching import Params
from postprocess import generate_confidence_scores, generate_confidence_score_plots # 引入信心分數和信心圖生成工具

# 將一對一的氨基酸三字母縮寫轉為單字母縮寫
one_to_three_amino ={'V':'VAL', 'I':'ILE', 'L':'LEU', 'E':'GLU', 'Q':'GLN',
'D':'ASP', 'N':'ASN', 'H':'HIS', 'W':'TRP', 'F':'PHE', 'Y':'TYR',  
'R':'ARG', 'K':'LYS', 'S':'SER', 'T':'THR', 'M':'MET', 'A':'ALA',
'G':'GLY', 'P':'PRO', 'C':'CYS'}

# 定義氨基酸對應的標籤
residue_label = {
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'C': 4,
    'Q': 5,
    'E': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19,
}

# 訓練過程中的氨基酸概率
aa_probs_train = [0.07943948021002176, 0.0497783411606611, 0.04433904104844631, 0.05350186331961819, 0.022626103099067648, 0.03937526841500076, 0.05627820955072524, 0.07217724394940638, 0.01882610864053863, 0.06108044830500256, 0.09799918263302994, 0.05331463086876411, 0.021703725253868638, 0.045350220966155465, 0.041743693113336935, 0.06416670130086032, 0.058655985481346026, 0.012496017067730628, 0.03598361109956638, 0.071164124516853]

# 初始化字典和變量
exclude_states = list() # 用來存儲排除的狀態
chain_id_states = dict() # 用來存儲鏈ID狀態
hmm_probability = list() # 存儲HMM的概率
seq_key_list = list() # 存儲序列鍵
chains_sequence_dict = dict() # 存儲鏈的序列
seq_list = list() # 存儲序列列表
chains_sec_sequence_dict = dict() # 存儲鏈的二級結構序列
transition_dic = dict() # 存儲轉移字典
transition_dic_c = dict()
transition_dic_n = dict()
hmm_dic = dict() # 存儲HMM字典
chain_list = list() # 存儲鏈列表
cord_idx_prob_dict = dict() # 存儲坐標索引和概率字典

chain_count = 0

# 記錄開始時間
start_time = time.time()

# 讀取HMM轉移矩陣文件、HMM文件及CA概率文件，並將它們加載到相應的字典中
def load_data(trans_file, hmm_file, save_ca_probs, trans_file_c, trans_file_n): #
    hmm_count = 0
    trans_count_ca = 0
    trans_count_c = 0
    trans_count_n = 0
    ca_prob_count = 0
    
    # creating key-value pair for hmm_file (讀取HMM檔案並創建key-value對)
    with open(hmm_file, 'r') as h_file:
        for line in h_file:
            hmm_count += 1
            h = line.strip()
            h = h.split()
            h_value = int(h[1].replace("\n", ""))
            hmm_dic[f'{h[0]}_{hmm_count}'] = h_value
        
    # creating key-value pair for trans_file (讀取轉移檔案並創建key-value對)
    with open(trans_file, 'r') as t_file:
        for line in t_file:
            t = line.replace("\n", "")
            transition_dic[trans_count_ca] = t
            trans_count_ca += 1
    
    # 讀取CA概率檔案
    with open(save_ca_probs, 'r') as ca_prob_f:
        for line in ca_prob_f:
            p = line.replace("\n", "")
            cord_idx_prob_dict[ca_prob_count] = p
            ca_prob_count += 1
    
    # 同理，產生 C 的轉移字典
    with open(trans_file_c, 'r') as t_file_c:  # trans_file_c 為 C 的檔案
        for line in t_file_c:
            t = line.replace("\n", "")
            transition_dic_c[trans_count_c] = t
            trans_count_c += 1

    # 同理，產生 N 的轉移字典
    with open(trans_file_n, 'r') as t_file_n:  # trans_file_n 為 N 的檔案
        for line in t_file_n:
            t = line.replace("\n", "")
            transition_dic_n[trans_count_n] = t
            trans_count_n += 1
            
######################## 進行原子之間的座標配對

def match_atoms(transition_dic, transition_dic_n, transition_dic_c,max_dist_n_ca,max_dist_ca_c):
    """
    最近鄰配對：根據典型鍵長閾值，將 CA 原子與最近的 N、C 原子配對。
    參數：
      transition_dic     CA 坐標字典 {idx: "x y z", …}
      transition_dic_n   N  原子坐標字典
      transition_dic_c   C  原子坐標字典
      max_dist_n_ca      N–CA 最大允許距離
      max_dist_ca_c      CA–C 最大允許距離
    回傳：
      matching 字典，格式 {ca_index: (n_index, c_index), …}
    """
    # max_dist_n_ca=config_dict['CA_N_DIST']
    # max_dist_ca_c=config_dict['CA_C_DIST']
    
    # 1. 將三個字典轉成 Nx3 的座標陣列，方便計算距離
    def to_array(dic):
        arr = []
        for i in sorted(dic.keys()):
            xyz = [float(x) for x in dic[i].split() if x!='']  # 分割字串並轉型
            arr.append(xyz)
        return np.array(arr)

    coords_ca = to_array(transition_dic)   # α-碳 (CA) 的座標
    coords_n  = to_array(transition_dic_n) # 胺氮 (N) 的座標
    coords_c  = to_array(transition_dic_c) # 羰基碳 (C) 的座標

    # 2. 定義找最近鄰函式：若最小距離小於閾值，回傳索引，否則回 None
    def find_nn(src, targets, max_dist):
        dists = np.linalg.norm(targets - src, axis=1) # 計算 src 到所有 target 點的歐氏距離，結果是一個長度為 N 的向量
        idx   = np.argmin(dists) # 找出距離最小的那個索引（argmin 回傳最小值的位置）
        return idx if dists[idx] <= max_dist else None # 如果最小距離小於等於 max_dist，就回傳索引；否則回傳 None

    # 3. 針對每個 CA，分別找距離最近的 N 和 C
    matching = {}
    for i_ca, ca in enumerate(coords_ca):
        i_n = find_nn(ca, coords_n,  max_dist_n_ca)
        i_c = find_nn(ca, coords_c,  max_dist_ca_c)
        if i_n is not None and i_c is not None:
            matching[i_ca] = (i_n, i_c)  # 同時找到合理鄰居
    return matching

def save(save_filename, matching, neighbor_mode="ALL"):
    """
    修改說明：
    1. 讀取三個不同的轉移字典：transition_dic (CA 坐標)、transition_dic_c (C 坐標)、
       transition_dic_n (N 坐標)。
    2. 根據 HMM 的結果 (hmm_dic) 中，每個殘基的 CA 索引，先從 matching 拿到對應的 N/C 索引列表。
    3. 如果 neighbor_mode == "ALL"，寫入所有配對到的鄰居；如果 == 1，僅寫入最近的那一個。
    4. 若某個 CA 沒有任何 N/C 配對，也仍保留該 CA 的輸出。
    5. 原始的 ATOM 排版格式不變，保持 atom_serial 與 residue_number 管理。
    """
    atom_serial = 1
    residue_number = 1

    # 小函式：從座標字典裡取 xyz、轉 float 並四捨五入
    def get_xyz(dic, idx):
        parts = [p for p in dic[idx].split() if p!='']
        return [round(float(x), 3) for x in parts]

    # 先寫入作者資訊
    with open(save_filename, 'a') as fi:
        fi.write("Author Cryo2Struct\n")

    # 遍歷 HMM 結果
    for key, ca_index in hmm_dic.items():
        residue_letter = key.split("_")[0]
        residue_name   = one_to_three_amino[residue_letter]
        chain_id       = chain_list[residue_number - 1]

        # CA 一定要寫，先取 CA 座標
        if ca_index >= 0 and ca_index < len(transition_dic):
            x_ca, y_ca, z_ca = get_xyz(transition_dic, ca_index)
        else:
            x_ca = y_ca = z_ca = 0.0

        # 先拿到 raw_n_list, raw_c_list（可能為空 list）
        raw_n_list, raw_c_list = matching.get(ca_index, ([], []))
        # 型別檢查：如果不是 list/tuple，就包成 list
        n_list = raw_n_list if isinstance(raw_n_list, (list,tuple)) else [raw_n_list] if raw_n_list is not None else []
        c_list = raw_c_list if isinstance(raw_c_list, (list,tuple)) else [raw_c_list] if raw_c_list is not None else []

        # 如果只要最近 1 個，就切到只有第一個
        if neighbor_mode == 1:
            n_list = n_list[:1]
            c_list = c_list[:1]

        # 開始寫入 PDB
        with open(save_filename, 'a') as fi:
            # 寫入 CA（無論有無鄰居，都要保留）
            fi.write("ATOM")
            fi.write("  ")
            fi.write(str(atom_serial).rjust(5))
            fi.write("  ")
            fi.write("CA".ljust(4))
            fi.write(residue_name.rjust(3))
            fi.write(" ")
            fi.write(chain_id)
            fi.write(str(residue_number).rjust(4))
            fi.write("    ")
            fi.write(str(x_ca).rjust(8))
            fi.write(str(y_ca).rjust(8))
            fi.write(str(z_ca).rjust(8))
            fi.write(str(1.00).rjust(5))
            fi.write(str(0.00).rjust(5))
            fi.write("           ")
            fi.write("C".rjust(1))
            fi.write("  \n")
            atom_serial += 1

            # 有找到 N，就寫入所有或第一個
            for n_idx in n_list:
                if 0 <= n_idx < len(transition_dic_n):
                    x_n, y_n, z_n = get_xyz(transition_dic_n, n_idx)
                else:
                    x_n = y_n = z_n = 0.0
                fi.write("ATOM")
                fi.write("  ")
                fi.write(str(atom_serial).rjust(5))
                fi.write("  ")
                fi.write(" N".ljust(4))
                fi.write(residue_name.rjust(3))
                fi.write(" ")
                fi.write(chain_id)
                fi.write(str(residue_number).rjust(4))
                fi.write("    ")
                fi.write(str(x_n).rjust(8))
                fi.write(str(y_n).rjust(8))
                fi.write(str(z_n).rjust(8))
                fi.write(str(1.00).rjust(5))
                fi.write(str(0.00).rjust(5))
                fi.write("           ")
                fi.write("N".rjust(1))
                fi.write("  \n")
                atom_serial += 1

            # 有找到 C，就寫入所有或第一個
            for c_idx in c_list:
                if 0 <= c_idx < len(transition_dic_c):
                    x_c, y_c, z_c = get_xyz(transition_dic_c, c_idx)
                else:
                    x_c = y_c = z_c = 0.0
                fi.write("ATOM")
                fi.write("  ")
                fi.write(str(atom_serial).rjust(5))
                fi.write("  ")
                fi.write(" C".ljust(4))
                fi.write(residue_name.rjust(3))
                fi.write(" ")
                fi.write(chain_id)
                fi.write(str(residue_number).rjust(4))
                fi.write("    ")
                fi.write(str(x_c).rjust(8))
                fi.write(str(y_c).rjust(8))
                fi.write(str(z_c).rjust(8))
                fi.write(str(1.00).rjust(5))
                fi.write(str(0.00).rjust(5))
                fi.write("           ")
                fi.write("C".rjust(1))
                fi.write("  \n")
                atom_serial += 1

        residue_number += 1

# def save(save_filename):
#     count = 0
#     with open(save_filename, 'a') as fi:
#         fi.write("Author Cryo2Struct\n")
#     for key, value in hmm_dic.items():
#         atom = "CA"
#         residue_name =  one_to_three_amino[key.split("_")[0]]
#         cord_idx = value # gaps are not included
#         if cord_idx >= 0 and cord_idx < len(transition_dic):
#             xyz = transition_dic[cord_idx].split(" ")
#             while '' in xyz:
#                 xyz.remove("")
            
#             x = round(float(xyz[0]), 3)
#             y = round(float(xyz[1]), 3)
#             z = round(float(xyz[2]), 3)
#             with open(save_filename, 'a') as fi:
#                 fi.write('ATOM')
#                 fi.write('  ')
#                 fi.write(str(count).rjust(5))
#                 fi.write('  ')
#                 fi.write(atom.ljust(4))
#                 fi.write(residue_name.rjust(3))
#                 fi.write(' ')
#                 fi.write(f'{chain_list[count]}')
#                 fi.write(str(count).rjust(4))
#                 fi.write('    ')
#                 fi.write(str(x).rjust(8))
#                 fi.write(str(y).rjust(8))
#                 fi.write(str(z).rjust(8))
#                 fi.write(str(1.00).rjust(5))
#                 fi.write(str(0.00).rjust(5))
#                 fi.write('           ')
#                 fi.write(atom[0:1].rjust(1))
#                 fi.write('  ')
#                 fi.write('\n')
#             count += 1
        
        
# 生成發射矩陣 (Emission matrix)，將發射概率從文件讀入
def makeEmission(emission_file, length_coordinate_list):
    emi_matrix = np.zeros((length_coordinate_list, 20), dtype=np.double)
    with open(emission_file,"r") as emission_f:
        idx = 0
        for line in emission_f:
            vals = line.split()
            for l in range(len(vals)):
                emi_matrix[idx][l] = vals[l]
            idx += 1
    return emi_matrix

# 根據訓練的氨基酸概率調整發射矩陣，使用幾何平均來更新概率值。
def makeEmission_aa(emission_mat):
    for em in range(len(emission_mat)):
        for aa_em in range(len(emission_mat[em])):
            emission_mat[em][aa_em] = np.sqrt(np.double(emission_mat[em][aa_em] * aa_probs_train[aa_em])) # geometric mean
    return emission_mat

# 正規化距離矩陣，使每行的總和為1。
def normalize_sum(coordinate_distance_matrix):
    coordinate_distance_matrix = coordinate_distance_matrix / coordinate_distance_matrix.sum(axis=1, keepdims=True)
    return coordinate_distance_matrix

# 根據正態分佈計算坐標距離的概率密度。
def probability_density_function(coordinate_distance_matrix_dis, std_lambda):
    computed_mean = 3.8047179727719045 # 固定平均值 (Å)
    computed_std = 0.03622304 * std_lambda # 標準差 = 基底值 × std_lambda
    p_norm = scipy.stats.norm(computed_mean,computed_std)
    probability_density_matrix  = p_norm.pdf(coordinate_distance_matrix_dis) # type: ignore
    return probability_density_matrix

# 處理並過濾非標準的氨基酸觀察。
def make_standard_observations(chain_obser):
    observations = tuple(chain_obser.strip('\n'))
    non_standard_amino_acids = ['X', 'U', 'O']
    filtered_observations = list()
    for o in observations:
        if o in non_standard_amino_acids:
            # print(f" - Removed {o}")
            pass # 移除非標準氨基酸
        else:
            filtered_observations.append(o)
    filtered_observations = list(tuple(filtered_observations))
    seq_list.extend(filtered_observations)
    return filtered_observations

# 釋放某一條鏈使用的變數與記憶體，並列印目前記憶體使用狀況。
import gc
import psutil
import os

def release_chain_memory(var_dict=None, print_prefix=""):
    """
    Parameters:
    ----------
    var_dict : dict
        欲刪除的變數名稱字典（通常用 locals() 傳入）。
    print_prefix : str
        輸出訊息的 prefix，例如鏈的名稱。
    """
    if var_dict is not None:
        for var_name in list(var_dict):
            if not var_name.startswith("__") and not callable(var_dict[var_name]):
                try:
                    del var_dict[var_name]
                except:
                    pass

    gc.collect()

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2  # MB
    print(f"🧹 {print_prefix} memory cleaned. Current usage: {mem:.2f} MB")




# 使用Viterbi算法進行鏈的比對。這部分通過C++實現。
def run_vitebi(key_idx, chain_observations, transition_matrix, emission_matrix, states, initial_matrix, config_dict, save_ca_probs, emission_matrix_dl):
    """
    # 1. 轉換 Python 的資料為 ctypes 格式，以便傳入 C++ shared lib
        #   - transition_matrix_log: 轉為 log 機率形式
        #   - emission_matrix_log, initial_matrix_log 也是 log
        #   - 所有陣列展平成一維陣列供 C++ 使用
    # 2. 載入 C++ 的 viterbi.so，設定參數格式
    # 3. 執行 Viterbi 解碼，回傳每個觀察值對應的最可能隱藏狀態（即：對應的坐標點）
    # 4. 結果存在 exclude_states 中，這表示這些點被選為最可能對應的狀態（即比對出來的模型 CA 座標）
    # 5. 若還有下一條鏈，遞迴呼叫 `execute()` 處理下一條序列
    # 6. 若是最後一條，則：
        #   - 儲存比對結果 (HMM)
        #   - 加載原子座標資料（CA、N、C）
        #   - 做原子間合理配對 (match_atoms)
        #   - 輸出結構為 PDB
        #   - 計算與輸出 confidence scores（包括回歸模型處理）
        #   - 繪製信心分數圖表
    """
    print(f"\n Cryo2Struct Alignment: Aligning Chain {seq_key_list[key_idx]}")

    chain_start_time = time.time()  # ⏱️ 開始計時

    chain_observations_np = np.array([residue_label[x] for x in chain_observations], dtype=np.int32)
    exclude_states_np = np.array(exclude_states, dtype=np.int32)

    transition_matrix_log = np.log(transition_matrix)
    emission_matrix_log = np.log(emission_matrix)
    initial_matrix_log = np.log(initial_matrix)


    states_len = len(states)
    exclude_arr_len = len(exclude_states_np)
    chain_arr_len = len(chain_observations_np)

    transition_arr = (ctypes.c_double * (states_len * states_len))()
    emission_arr = (ctypes.c_double * (states_len * 20))()
    initial_arr = (ctypes.c_double * len(initial_matrix_log))()
    exclude_arr = (ctypes.c_int * len(exclude_states_np))()
    chain_arr = (ctypes.c_int * len(chain_observations_np))()

    # 填充矩陣  
    for i in range(states_len):
        for j in range(states_len):
            transition_arr[i*states_len + j] = transition_matrix_log[i,j]
    
    for i in range(states_len):
        for j in range(20):
            emission_arr[i*20 + j] = emission_matrix_log[i,j]
    

    for i in range(len(initial_matrix_log)):
        initial_arr[i] = initial_matrix_log[i]


    for i in range(len(exclude_states_np)):
        exclude_arr[i] = exclude_states_np[i]

    for i in range(len(chain_observations_np)):
        chain_arr[i] = chain_observations_np[i]

    

    # Load the C++ shared library (載入C++的viterbi庫)
    viterbi_algo_path = os.path.abspath(config_dict['input_data_dir'])
    viterbi_algo_path = os.path.dirname(viterbi_algo_path)
    lib = ctypes.cdll.LoadLibrary(f'{viterbi_algo_path}/viterbi/viterbi.so')

    # Define the C++ wrapper function
    wrapper_function = lib.viterbi_main

    # Define the argument types for the wrapper function
    wrapper_function.argtypes = [
    ctypes.POINTER(ctypes.c_int), # chain_o
    ctypes.c_int, # chain_o_len
    ctypes.c_int, # num_states
    ctypes.POINTER(ctypes.c_double), # transition_matrix_log
    ctypes.POINTER(ctypes.c_double), # emission_matrix_log
    ctypes.POINTER(ctypes.c_double), # initial_matrix_log
    ctypes.POINTER(ctypes.c_int), # exclude_arr
    ctypes.c_int, # exclude_arr_len
    ]

    wrapper_function.restype = ctypes.POINTER(ctypes.c_int)

    v_start = time.time()
    results = wrapper_function(chain_arr, chain_arr_len, states_len, transition_arr, emission_arr, initial_arr, exclude_arr, exclude_arr_len)
    v_end = time.time()
    print(f"🔍 C++ Viterbi run time: {v_end - v_start:.2f} sec", flush=True)

    # results = wrapper_function(chain_arr, chain_arr_len, states_len, transition_arr, emission_arr, initial_arr, exclude_arr, exclude_arr_len)
    observation_length_for_c = len(chain_observations)
    exclude_state_from_c = np.ctypeslib.as_array(results, shape=(observation_length_for_c,))
    exclude_states.extend(exclude_state_from_c)

    # 新增釋放記憶體
    release_chain_memory(var_dict=locals(), print_prefix=f"Chain {seq_key_list[key_idx]}")


    chain_end_time = time.time()  # ⏱️ 結束計時
    runtime_sec = chain_end_time - chain_start_time
    print(f"⏱️ Chain  {seq_key_list[key_idx]} alignment took {runtime_sec:.2f} seconds.")


    # 將結果進行處理並保存PDB文件
    key_idx += 1
    if key_idx < len(seq_key_list):
        execute(key_idx=key_idx, states=states,transition_matrix=transition_matrix, emission_matrix=emission_matrix, config_dict=config_dict, save_ca_probs=save_ca_probs, emission_matrix_dl=emission_matrix_dl)
    else:
        # 清理並保存最終結果
        cord_file_ca = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_ca.txt"
        cord_file_c = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_c.txt"
        cord_file_n = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_n.txt"
        
        
        hmm_out_save_file = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_hmm_{config_dict['use_sequence']}.txt"
        save_pdb_file = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cryo2struct_{config_dict['use_sequence']}.pdb"
        
        conf_score_pdb_file = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cryo2struct_{config_dict['use_sequence']}_conf_score.pdb"
        
        save_confidence_score = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cryo2struct_confidence_scores.csv"
        save_prob_score = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cryo2struct_prob_scores.csv"
        save_conf_score_plot = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cryo2struct_conf_scores.png"
        
        # 清除舊文件
        if os.path.exists(save_confidence_score):
            os.remove(save_confidence_score)

        if os.path.exists(save_prob_score):
            os.remove(save_prob_score)

        if os.path.exists(hmm_out_save_file):
            os.remove(hmm_out_save_file)

        if os.path.exists(save_pdb_file):
            os.remove(save_pdb_file)

        if os.path.exists(conf_score_pdb_file):
            os.remove(conf_score_pdb_file)

        if os.path.exists(save_conf_score_plot):
            os.remove(save_conf_score_plot)
        
        # 保存HMM輸出
        hmm_outs = open(hmm_out_save_file, 'a')
        for i in range(len(exclude_states)):     
            print(f"{seq_list[i]}\t{exclude_states[i]}",file=hmm_outs)
        hmm_outs.close()
        
        # 加載數據並保存PDB
        # load_data(trans_file=cord_file_ca, hmm_file=hmm_out_save_file, save_ca_probs=save_ca_probs)
        load_data(trans_file=cord_file_ca, hmm_file=hmm_out_save_file, save_ca_probs = save_ca_probs, trans_file_c=cord_file_c, trans_file_n=cord_file_n)
        
        # 2. 建立所有合理鄰居配對
        ## 簡陋版原子匹配
        # print(f"\n old_bond_match\n " )
        # matching = match_atoms(transition_dic,transition_dic_n,transition_dic_c,max_dist_n_ca=config_dict['CA_N_DIST'],max_dist_ca_c=config_dict['CA_C_DIST'])
        
        ## 成本版原子匹配
        print(f"\n new_bond_match\n " )
        bm_cfg = config_dict['bond_matching']
    
        if isinstance(bm_cfg, dict):
            # 從 YAML 來的是 dict，把它變成 Params dataclass
            bm_p = Params(**bm_cfg)

        matching = bond_matching.bond_match(transition_dic,transition_dic_n,transition_dic_c, bm_p,return_matching=True)
        
        # 3a. 輸出所有鄰居
        # save(save_filename=save_pdb_file, matching=matching, neighbor_mode="ALL")

        # 3b. 僅輸出最近一個鄰居
        save(save_filename=save_pdb_file, matching=matching, neighbor_mode=1)
        # save(save_filename=save_pdb_file)
        
        
        print("\n Cryo2Struct Alignment: Total modeled residues:", len(set(exclude_states)))
        end_time = time.time()
        runtime_seconds = end_time - start_time
        runtime_minutes = runtime_seconds / 60
        print(f"Cryo2Struct Alignment: Run time {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")

        ######## clean up:  清理掉過程文件 這部分應該註解要 才可以輸出他的資料
        # map_directory_path = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}"
        
        # if os.path.exists(f"{map_directory_path}/{config_dict['density_map_name']}_amino_predicted.mrc"):
        #     os.remove(f"{map_directory_path}/{config_dict['density_map_name']}_amino_predicted.mrc")
            
        # if os.path.exists(f"{map_directory_path}/{config_dict['density_map_name']}_atom_predicted.mrc"):
        #     os.remove(f"{map_directory_path}/{config_dict['density_map_name']}_atom_predicted.mrc")
        
        # files_to_delete = glob.glob(os.path.join(map_directory_path, f"*.txt"))
        # for f in files_to_delete:
        #     os.remove(f)
            
        # 生成並保存信心分數、概率、圖表
        print("Cryo2Struct: Finished!\n")
        ami_list = list()
        ca_list = list()
        seq_list_conf = list()

        for k,v in hmm_dic.items():
            amino = k.split("_")[0]
            seq_list_conf.append(amino)
            ami=  residue_label[amino]
            ami_list.append(emission_matrix[v][ami])
            ca_list.append(float(cord_idx_prob_dict[v]))

        # 保存信心分數及輸出csv
        generate_confidence_scores.res_prob_score_files(save_prob_score_file=save_prob_score, seq_list=seq_list, 
                                                        seq_list_conf=seq_list_conf, ca_list=ca_list, ami_list=ami_list)
        
        trained_regression_model_aa = f"{config_dict['confidence_score_models']}/aa_regression_model.pkl"
        trained_regression_model_ca = f"{config_dict['confidence_score_models']}/ca_regression_model.pkl"
        
        generate_confidence_scores.gen_conf_scores(prob_scores=save_prob_score, save_path=save_confidence_score,
                                                   trained_regression_model_aa=trained_regression_model_aa, 
                                                   trained_regression_model_ca=trained_regression_model_ca)
        
        # 保存最後的pdb檔案
        generate_confidence_score_plots.save_scores_to_pdb(conf_score_file=save_confidence_score, 
                                                       input_pdb_file=save_pdb_file, 
                                                       output_pdb_file=conf_score_pdb_file)
        
        # 繪製點散圖
        avg_ca_conf, avg_aa_conf = generate_confidence_score_plots.generate_plot(conf_score_file=save_confidence_score, plot_filename=save_conf_score_plot)
        
        
        print(f"+ Cryo2Struct Outputs: ")
        print(f"Average carbon-alpha and amino acid-type confidence score are {avg_ca_conf} and {avg_aa_conf}, respectively.")
        print("Modeled Structure saved path:")
        print(f"- {conf_score_pdb_file}")
        print("Confidence Score csv file save path:") 
        print(f"- {save_confidence_score}")
        print("Confidence Score plot saved path:")
        print(f"- {save_conf_score_plot}")

        exit()

def execute(key_idx, states, transition_matrix, emission_matrix, config_dict, save_ca_probs, emission_matrix_dl):
    """
    # - 取得目前 key_idx 對應的 chain
    # - 將 chain 轉為觀察序列 (標準化成 label index)
    # - 計算每個 state 的初始機率 P(state_i | first_observation)
    # - 呼叫 run_vitebi 執行實際比對

    """
    chain_sequence = chains_sequence_dict[seq_key_list[key_idx]]
    chain_observations = make_standard_observations(chain_obser=chain_sequence)
    initial_hidden_pobabilities = np.zeros((len(states)), dtype=np.double)
    observation_seq_first_amino_count = 0
    for i_c in range(len(states)):
        observation_seq_first_amino_count += emission_matrix[i_c][residue_label[chain_observations[0]]] 
    for i_c in range(len(states)):
        initial_hidden_pobabilities[i_c] = emission_matrix[i_c][residue_label[chain_observations[0]]] / observation_seq_first_amino_count
    run_vitebi(key_idx=key_idx, chain_observations=chain_observations ,transition_matrix=transition_matrix, emission_matrix=emission_matrix, 
               states=states, initial_matrix=initial_hidden_pobabilities, config_dict=config_dict, save_ca_probs=save_ca_probs, emission_matrix_dl=emission_matrix_dl)



def main(coordinate_file, emission_file, config_dict, save_ca_probs):
    """
    # 1. 讀取 .fasta，取得蛋白質序列（可來自原始或pdb提取）
    # 2. 讀入坐標（Carbon-Alpha原子），並建立距離矩陣
    # 3. 將距離矩陣套用機率密度函數 + 正規化 → 得 transition matrix
    # 4. 讀入 emission 機率（預測的殘基類型機率）→ 得 emission matrix
    # 5. 將 emission matrix 轉為 20 類型格式 (amino acid)
    # 6. 建立 HMM 的 state 空間（狀態數 = 坐標點數 = CA原子數）
    """

    FASTA_start_time = time.time()
    # 讀取 FASTA 序列
    fasta_file = [f for f in os.listdir(f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}") if f.endswith(".fasta")]
    fasta_file.sort()
    
    
    if config_dict['use_sequence'] == "full":
    # 使用完整的蛋白質序列來進行比對
        sequence_file = fasta_file[0]
        print("\nCryo2Struct Alignment: Running with full fasta sequence")
    
    else:
        pdb_name = fasta_file[0].split(".")[0]
        pdb_name = pdb_name.split("_")[0]
        pdb_file_p = f"{pdb_name.lower()}.pdb"
        pdb_file_dir_p = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{pdb_file_p}"
        
        if os.path.exists(pdb_file_dir_p):
            pdb_file_dir = pdb_file_dir_p
        else:
            pdb_file_e = f"{pdb_name.lower()}"
            pdb_file_dir_ent = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{pdb_file_e}.ent"
            pdb_file_dir_pdb = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{pdb_file_e}.pdb"
            if os.path.exists(pdb_file_dir_ent):
                pdb_file_dir = pdb_file_dir_ent
            elif os.path.exists(pdb_file_dir_pdb):
                pdb_file_dir = pdb_file_dir_pdb

        # print(pdb_file_dir)
        atomic_seq_chain_file = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/atomic_seq_chain.fasta"
        atomic_seq_file = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/atomic_seq.fasta"
        extract_seq_from_pdb.extract_seq(pdb_file_dir, atomic_seq_chain_file, atomic_seq_file)
    
        if config_dict['use_sequence'] == "atomic_no_chain":
            sequence_file = atomic_seq_file 
            print("Running with ATOMIC NO CHAIN SEQUENCE")
        elif config_dict['use_sequence'] == "atomic_chain":
            sequence_file = atomic_seq_chain_file
            print("Running with CHAIN SEQUENCE")

    
    # read the coordinate file and append them to list
    coordinate_list = list()
    with open(coordinate_file,"r") as coordinate_f:
        for line in coordinate_f:
            x_y_z = [float(x) for x in line.split()]
            coordinate_list.append(x_y_z)
    
    # create a numpy array filled with zeros with size as coordinate file
    length_coordinate_list = len(coordinate_list)
    coordinate_distance_matrix = np.zeros((length_coordinate_list, length_coordinate_list), dtype=np.double)
    
    # compute distance between each carbon alpha to other and put into distance matrix
    
    # # 1. 取得長度 (LEN)
    length_coordinate_list = len(coordinate_list)
    # print(f"--- 基本資訊 ---")
    # print(f"Alpha 碳原子數量 (LEN): {length_coordinate_list}")
    # print(f"計算總次數 (LEN x LEN): {length_coordinate_list * length_coordinate_list}")
    # print("-" * 30)    

    for carbon_alpha in range(length_coordinate_list):
        for carbon_alpha_next in range(length_coordinate_list):
            coordinate_distance_matrix[carbon_alpha][carbon_alpha_next] = np.linalg.norm(np.array(coordinate_list[carbon_alpha]) - np.array(coordinate_list[carbon_alpha_next]))

    coordinate_distance_matrix_dis = coordinate_distance_matrix
    coordinate_distance_matrix = probability_density_function(coordinate_distance_matrix_dis, config_dict['std_lambda'])
    coordinate_distance_matrix += 1e-20
    coordinate_distance_matrix = normalize_sum(coordinate_distance_matrix)
    
    emission_matrix = makeEmission(emission_file, length_coordinate_list)
    emission_matrix += 1e-20
    emission_matrix = normalize_sum(emission_matrix)
    emission_matrix_dl = emission_matrix
    emission_matrix_aa = makeEmission_aa(emission_matrix)
    emission_matrix_aa = normalize_sum(emission_matrix_aa)

    assert coordinate_distance_matrix.shape == (length_coordinate_list, length_coordinate_list)
    for row in range(len(coordinate_distance_matrix[0])):
        assert abs(sum(coordinate_distance_matrix[row]) - 1) < 0.0001, f'Row {row} does not sum to 1 in transition matrix'
    
    assert emission_matrix_aa.shape == (length_coordinate_list, 20)
    for row in range(length_coordinate_list):
        assert abs(sum(emission_matrix_aa[row]) - 1) < 0.0001, f'Row {row} does not sum to 1 in emission matrix AMINO'

    states= list(tuple(idx for idx in range(length_coordinate_list)))

    # 程式打開選定的 FASTA 檔案，並將內容讀入 seq_lines 變數中。
    with open(os.path.join(config_dict['input_data_dir'], config_dict['density_map_name'], sequence_file),"r") as seq_f:
        seq_lines = seq_f.readlines()
    
    # org_解析 FASTA 序列
    # for seq_contents in range(0,len(seq_lines),2):
    #     seq_c = seq_lines[seq_contents]
    #     seq_c = seq_c.split("|")[1] 
    #     seq_c  = seq_c.split(" ") 
    #     seq_c = seq_c[1:]
    #     for seq_chain in seq_c:
    #         seq_key = seq_chain.replace(",","").strip('\n')
    #         seq_key_list.append(seq_key)
    #         chains_sequence_dict[seq_key] = seq_lines[seq_contents + 1].strip('\n')
    # for ke, va in chains_sequence_dict.items():
    #     length_va = len(va)
    #     chain_list.extend(ke*length_va)

    # fix_解析 FASTA 序列
    for seq_contents in range(0,len(seq_lines),2):
        seq_c = seq_lines[seq_contents]
        seq_c = seq_c.split("|")[1] 
        seq_c = re.sub(r'^[Cc]hains?\s*', '', seq_c) # 去掉開頭的 "Chains " 或 "Chain "
        seq_c = [seg.strip() for seg in seq_c.split(',')]

        for seq_chain in seq_c:
            seq_key = seq_chain.replace(",","").strip('\n')
            seq_key_list.append(seq_key)
            chains_sequence_dict[seq_key] = seq_lines[seq_contents + 1].strip('\n')

    for ke, va in chains_sequence_dict.items():
        clean_ke = ke[0]
        length_va = len(va)
        chain_list.extend(clean_ke*length_va)

    FASTA_end_time = time.time()
    print(f"[Time] : {FASTA_end_time - FASTA_start_time:.2f} seconds")
    
    # ======================================================================
    # 直接手動輸出聚類結果 PDB (不依賴全局 save 函式)
    # ======================================================================
    print("\n🚀 正在手動輸出聚類結果 PDB (確保座標不為 0)...")
    
    pre_cluster_pdb = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_clusters_pre_hmm.pdb"
    
    # 讀取 C 和 N 的檔案以確保配對完整 (如果有的話)
    def load_txt_coords(filename):
        res = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                for line in f:
                    parts = [float(x) for x in line.split() if x.strip()]
                    if parts: res.append(parts)
        return res

    raw_ca = coordinate_list # 使用 main 已經讀好的
    raw_c  = load_txt_coords(f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_c.txt")
    raw_n  = load_txt_coords(f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_n.txt")

    with open(pre_cluster_pdb, 'w') as f_out:
        f_out.write("Author Cryo2Struct - Cluster Only\n")
        atom_idx = 1
        for i in range(len(raw_ca)):
            res_num = i + 1
            # 寫入 CA
            x, y, z = raw_ca[i]
            f_out.write(f"ATOM  {atom_idx:>5}  CA  VAL U{res_num:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C\n")
            atom_idx += 1
            
            # 寫入 N (如果有的話)
            if i < len(raw_n):
                nx, ny, nz = raw_n[i]
                f_out.write(f"ATOM  {atom_idx:>5}  N   VAL U{res_num:>4}    {nx:>8.3f}{ny:>8.3f}{nz:>8.3f}  1.00  0.00           N\n")
                atom_idx += 1
                
            # 寫入 C (如果有的話)
            if i < len(raw_c):
                cx, cy, cz = raw_c[i]
                f_out.write(f"ATOM  {atom_idx:>5}  C   VAL U{res_num:>4}    {cx:>8.3f}{cy:>8.3f}{cz:>8.3f}  1.00  0.00           C\n")
                atom_idx += 1
        f_out.write("END\n")

    print(f"✅ 聚類後的 PDB 已生成至: {pre_cluster_pdb}")
    # ======================================================================

    key_idx = 0
    print("Cryo2Struct Alignment: HMM Construction Complete!")
    print("fasta_seq_key_list :",seq_key_list)
    
    # 讀取 cluster_transition_ca.txt 確認 ca 數量
    with open(f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_ca.txt") as f:
        ca_centers = [l for l in f if l.strip()]
    num_ca = len(ca_centers)
    print(f"CA 聚類後數量: {num_ca}")


    # 你程式里已經把序列讀到 chains_sequence_dict
    # 2a. 单条链长度
    for chain_id, seq in chains_sequence_dict.items():
        print(f"鏈 {chain_id} 的殘基數量: {len(seq)}")

    # 2b. 或所有链的残基总数
    total_residues = sum(len(seq) for seq in chains_sequence_dict.values())
    print(f"所有鏈殘基總數: {total_residues}")

    if num_ca >= total_residues:
        print("✅ CA 聚類數量 ≥ 殘基數量，可以嘗試一一對應。")
    else:
        print("❌ CA 聚類數量 < 殘基數量，可能會有殘基配不到位置。")

    execute(key_idx=key_idx, states=states,transition_matrix=coordinate_distance_matrix, emission_matrix=emission_matrix_aa, config_dict =config_dict, save_ca_probs=save_ca_probs, emission_matrix_dl=emission_matrix_dl)
    exit()
