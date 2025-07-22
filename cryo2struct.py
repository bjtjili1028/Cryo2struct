"""
Created on 04 Sep 2023 06:16 AM
@author: nabin

"""

import time  # 用於計算執行時間
import argparse  # 用於處理命令行參數 
import yaml  # 用於讀取yml格式的配置文件 
import os  # 用於處理檔案和目錄 
import shutil  # 用於刪除目錄和文件 
import threading  # 用於創建和管理線程

from utils import get_probs_cords_from_atom_amino, clustering_centroid, grid_division, clustering_centroid_for_c_n  # 自定義的工具函數 
from viterbi import alignment  # 用於viterbi演算法的對齊 
import subprocess  # 用於執行外部命令 

import warnings  # 用於警告控制 
warnings.filterwarnings("ignore")  # 忽略警告 

# 取得當前腳本的目錄路徑
script_dir = os.path.dirname(os.path.abspath(__file__))

# 配置文件的路徑
config_file_path = f"{script_dir}/config/arguments.yml"
COMMENT_MARKER = '#'  # 註解的標誌

# 解析命令行參數 
def parse_arguments():
    parser = argparse.ArgumentParser()
    # 解析配置文件的參數
    parser.add_argument('--config', type=argparse.FileType(mode='r'),
                        default=config_file_path)
    parser.add_argument('--density_map_name', type=str) # 密度圖名稱
    
    return parser.parse_args()  # 返回解析後的參數

# 處理命令行參數
def process_arguments(args):
    if args.config is not None:
        # 讀取配置文件並過濾掉註解行
        config_dict = yaml.safe_load(args.config) # 讀取
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith(COMMENT_MARKER)} # 過濾
        args.config = args.config.name # 配置文件的路徑
    else:
        config_dict = dict() # 如果沒有提供配置文件，則創建一個空字典
    
    # 如果提供了密度圖名稱，將其添加到配置字典
    if args.density_map_name is not None:
        config_dict['density_map_name'] = args.density_map_name
    return config_dict

# 刪除指定的目錄
def delete_directory(directory_path):
    shutil.rmtree(directory_path)
           
# 進行預測
def make_predictions(config_dict):
    # 分割網格
    start_time = time.time()
    grid_division.create_subgrids(input_data_dir=config_dict['input_data_dir'], density_map_name=config_dict['density_map_name'])
    end_time = time.time()
    print(f"\nCryo2Struct DL: Grid Division Complete! \n[Time] : {end_time - start_time:.2f} seconds")

    # print("\nCryo2Struct DL: Grid Division Complete!")
    # 設定密度圖的目錄路徑
    density_map_dir = os.path.join(config_dict['input_data_dir'],config_dict['density_map_name'])
    density_map_split_dir = os.path.join(density_map_dir, f"{config_dict['density_map_name']}_splits")
    # 設定腳本名稱和檢查點名稱
    script_name = ['../../infer/atom_inference.py', '../../infer/amino_inference.py']
    checkpoint_name = ['atom_checkpoint', 'amino_checkpoint']

    # 執行每一個推斷腳本
    for s in range(len(script_name)):
        # 從cmd輸入對應指令 "python3 cryo2struct.py --density_map_name 34610"
        start_time = time.time()

        cmd = ['python3', script_name[s], density_map_split_dir, str(config_dict['input_data_dir']),
               str(config_dict['density_map_name']),  str(config_dict[checkpoint_name[s]]) , config_dict['infer_run_on'], str(config_dict['infer_on_gpu'] )]
        try:
            # 執行外部命令
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            stdout = result.stdout
            stderr = result.stderr
            return_code = result.returncode
            if return_code == 0:
                print(f"Cryo2Struct DL: Prediction {s + 1} / {len(script_name)} Complete!")
                # print(stdout)
            else:
                print(f"Cryo2Struct Deep Learning Block failed with exit code {return_code}.")
                print("Standard Error:")
                print(stderr)
        except subprocess.CalledProcessError as e:
            print(f"Cryo2Struct Deep Learning Block failed with exit code {e.returncode}.")
            print("Standard Error:")
            print(e.stderr)
        except Exception as e:
            print(f"An error occurred in Cryo2Struct Deep Learning Block: {str(e)}")
        
        end_time = time.time()
        print(f"[Time] : {end_time - start_time:.2f} seconds")

    # 創建新線程來刪除目錄（非同步操作）
    delete_thread = threading.Thread(target=delete_directory, args=(density_map_split_dir,))
    delete_thread.start() # runs in background to delete the grid division directory
    delete_thread1 = threading.Thread(target=delete_directory, args=(f"{density_map_dir}/lightning_logs",))
    delete_thread1.start()

# 提取基於atom和amino的概率和坐標
def extract_probs_cords_from_atom_amino(config_dict):
    
    # 讀取模型輸出檔案
    probability_file_atom = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_atom.txt" # comes from atom_inference.py
    
    probability_file_atom_spilt_ca = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_spilt_ca_prob.txt" #修改成僅讀取切割後的ca原子
    
    probability_file_amino = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino.txt" # comes from amino_inference.py
    
    # 輸出共同原子的檔案(胺基酸和原子進行比對後的數據)
    probability_file_amino_atom_common_emi = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino_atom_common_emi.txt" # save common amino and atom
    probability_file_amino_common_emi = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino_emi.txt" # save amino probability as emission
    probability_file_amino_atom_common_ca_prob = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino_atom_common_ca_prob.txt" # save common amino and atom (atom prob)
    save_cords = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_coordinates_ca.txt" # save cords as transition matrix

    # 輸出三種骨幹原子
    split_output_ca  = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_spilt_ca_prob.txt" # save ca atom prob
    split_output_n = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_spilt_n_prob.txt" # save n atom prob
    split_output_c = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_spilt_c_prob.txt" # save c atom prob
    
    # 刪除已存在的文件
    if os.path.exists(save_cords):
        os.remove(save_cords)
    
    if os.path.exists(probability_file_amino_atom_common_emi):
        os.remove(probability_file_amino_atom_common_emi)

    if os.path.exists(probability_file_amino_common_emi):
        os.remove(probability_file_amino_common_emi)
    
    ############# 分割三種骨幹原子 #####################
    get_probs_cords_from_atom_amino.split_atom_file(
                    probability_file_atom=probability_file_atom, 
                    split_output_ca = split_output_ca, 
                    split_output_n = split_output_n, 
                    split_output_c = split_output_c,
                    ca_threshold = config_dict['threshold'],
                    mode=config_dict['split_mod'] # mod=1是取最大機率值  mod=2是取0.4當作標準 
                    ) 
    
    # 計算並保存概率和坐標    
    get_probs_cords_from_atom_amino.get_joint_probabity_common_threshold(
                    probability_file_atom=probability_file_atom_spilt_ca,  # 原本使用 probability_file_atom
                    probability_file_amino_atom_common = probability_file_amino_atom_common_emi, 
                    probability_file_amino = probability_file_amino, 
                    s_c=save_cords, threshold = config_dict['threshold'],
                    probability_file_amino_atom_common_ca_prob = probability_file_amino_atom_common_ca_prob)

# 執行聚類來準備發射和轉換矩陣
def cluster_emission_transition(config_dict):
    save_cords, save_probs_aa, save_ca_probs= clustering_centroid.main(config_dict)
    combined_output_file_c, coords_output_file_c, save_cords_c,combined_output_file_n, coords_output_file_n, save_cords_n = clustering_centroid_for_c_n.main(config_dict) 
    return save_cords, save_probs_aa, save_ca_probs,combined_output_file_c, coords_output_file_c, save_cords_c,combined_output_file_n, coords_output_file_n, save_cords_n

# 主函數，執行整個流程
def main():
    args = parse_arguments() # 解析命令行參數
    config_dict = process_arguments(args) # 處理命令行參數
    print("\n##############- Cryo2Struct -##############")
    print("\nRunning with below configuration: ")
    
    # 打印配置信息
    for key,value in config_dict.items():
        print("%s : %s"%(key, value))
    print("\n- This might take a bit. Time for a coffee break, maybe! -")
    
    # 進行預測
    make_predictions(config_dict)
    
    cluster_start_time = time.time()
    # preparing for HMM model  (提取概率和坐標)
    extract_probs_cords_from_atom_amino(config_dict)
    
    # clustering and preparing emission and transition matrix (聚類處理)
    coordinate_file, emission_file, save_ca_probs,combined_output_file_c, coords_output_file_c, save_cords_c,combined_output_file_n, coords_output_file_n, save_cords_n = cluster_emission_transition(config_dict)
    # coordinate_file, emission_file, save_ca_probs = cluster_emission_transition(config_dict)
    cluster_end_time = time.time()
    runtime_sec = cluster_end_time - cluster_start_time
    print(f"\nCryo2Struct Clustering Finished {runtime_sec:.2f} seconds.")

    # run viterbi algorithm (執行Viterbi算法)
    alignment.main(coordinate_file, emission_file, config_dict, save_ca_probs)

# 程式入口，執行主函數
if __name__ == "__main__":
    main()
