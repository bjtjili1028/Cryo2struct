"""
Created on 18 April 2023 1:23 AM
@author: nabin

是計算ca和胺基酸類型都有的座標位置

"""
import math
import ast
import os

# 全局字典，用來存儲氨基酸、二級結構和原子對應的概率
prob_dic_aa = dict()
prob_dic_sec = dict()
prob_dic_atom = dict()

# 定義一個點(Point)類別，表示三維空間中的一個點
class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# 計算兩個點之間的歐氏距離
def distance(p1, p2):
    dis =  math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
    return dis

# 根據給定的閾值（thres）將點集進行聚類
def create_clusters(points, thres):
    clusters = [] # 存儲聚類結果
    while points:
        # 隨機選擇一個點並將其分配到新的一個聚類中
        # Select a point randomly and assign it to a new cluster
        cluster = [points.pop(0)]
        # 遍歷其餘的點，若與當前聚類中心的距離小於閾值，則加入該聚類
        # Iterate through the rest of the points and add them to the cluster if they are within the threshold distance
        i = 0
        while i < len(points):
            if distance(cluster[0], points[i]) <= thres:
                cluster.append(points.pop(i)) # 加入聚類
            else:
                i += 1
        # 將完成的聚類加入結果
        clusters.append(cluster)
    return clusters

# 計算每個聚類的質心並將結果寫入文件
def centroid(file, save_cords, save_probs_aa , thres, save_ca_probs):
# def centroid(file, save_cords, save_probs_aa , save_ca_probs):

    # Read the data from the file (從文件中讀取數據)
    points = []
    with open(file, 'r') as f:
        # Read all the lines in the file
        lines = f.readlines()

    # 將每個點的座標存入points列表
    for line in lines:
        vals = line.split(" ")
        for limiter in vals:
            if limiter == '':
                vals.remove(limiter)
        # 去除空白項
        vals = list(filter(lambda x: x != '', vals))
        # 創建Point對象並加入列表               
        points.append(Point(float(vals[0]), float(vals[1]), float(vals[2])))

    # Create the clusters
    # 根據閾值將點進行聚類
    clusters = create_clusters(points, thres=thres)
    # clusters = [[p] for p in points]
    
    # 打開結果文件以寫入結果
    with open(save_probs_aa,'w') as p:
        with open(save_cords, 'w') as f:
            with open(save_ca_probs, 'w') as a_p:
                for i, cluster in enumerate(clusters):
                    x_sum = 0
                    y_sum = 0
                    z_sum = 0
                    num_points = len(cluster)
                    collect_values = list()
                    collect_values_sec = list()
                    collect_values_atom = list()
                    for point in cluster:
                        x_sum += point.x
                        y_sum += point.y
                        z_sum += point.z
                        cords = (point.x, point.y, point.z) # 當前點的坐標
                        # 獲取該坐標對應的概率值
                        if cords in prob_dic_aa:
                            values = prob_dic_aa.get(cords)
                            collect_values.append(values) 
                            atom_values = prob_dic_atom.get(cords)
                            collect_values_atom.append(atom_values)
                        if cords in prob_dic_sec:
                            values = prob_dic_sec.get(cords)
                            collect_values_sec.append(values)
                    
                    # 計算聚類的均值
                    averages = list()
                    averages_atom = list()
                    for i in range(len(collect_values[0])):
                        total = sum(collect_values[j][i] for j in range(len(collect_values))) # 求和
                        average = total / len(collect_values)  # 計算均值
                        averages.append(average) 
                    averages = ' '.join(str(x) for x in averages)
                    for i in range(len(collect_values_atom[0])):
                        total = sum(collect_values_atom[j][i] for j in range(len(collect_values))) # 求和
                        average_atm = total / len(collect_values_atom)  # 計算均值
                        averages_atom.append(average_atm) 
                    averages_atom = ' '.join(str(x) for x in averages_atom)
                    
                    # 將結果寫入文件
                    print(averages, file=p)
                    print(averages_atom, file=a_p)   
                    
                    # 計算質心並寫入文件
                    x_avg = x_sum / num_points
                    y_avg = y_sum / num_points
                    z_avg = z_sum / num_points
                    
                    # 寫入質心坐標
                    print(f'{x_avg} {y_avg} {z_avg}', file=f)
                    
# 處理氨基酸的概率數據和原子數據，並將其保存到全局字典中
def proc_probabilities_aa(file, file_atom):
    # 讀取氨基酸的概率數據
    with open(file, 'r') as f:
        # Read all the lines in the file
        line = f.readline()
        while line:
            line_c = ast.literal_eval(line)
            key = tuple(line_c[0]) # 提取坐標作為鍵
            vals = line_c[1:] # 提取對應的概率值
            prob_dic_aa[key] = vals # 存儲到字典中
            line = f.readline()
    
    # 讀取原子數據的概率數據
    with open(file_atom, 'r') as f:
        line = f.readline()
        while line:
            line_c = ast.literal_eval(line)
            key = tuple(line_c[0]) # 提取坐標作為鍵
            vals = line_c[1:] # 提取對應的概率值
            prob_dic_atom[key] = vals # 存儲到字典中
            line = f.readline()

    
# 主函數，執行氨基酸預測過程
def main(config_dict):
    
    # 讀取檔案
    cord_data = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_coordinates_ca.txt"
    cord_probs_aa = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino_atom_common_emi.txt"
    cords_prob_atom = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_probabilities_amino_atom_common_ca_prob.txt" 
    
    # 存放檔案
    save_cords = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_ca.txt"
    save_probs_aa = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_emission_aa_ca.txt"
    save_ca_probs = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_ca_probs.txt"
    
    # 如果保存路徑已經存在，則刪除
    if os.path.exists(save_cords):
        os.remove(save_cords)

    if os.path.exists(save_probs_aa):
        os.remove(save_probs_aa)

    # 處理氨基酸和原子數據的概率
    proc_probabilities_aa(cord_probs_aa, cords_prob_atom)
    
    # 計算質心並保存結果
    centroid(cord_data, save_cords, save_probs_aa, config_dict['clustering_threshold'], save_ca_probs)
    # centroid(cord_data, save_cords, save_probs_aa, save_ca_probs)
    
    # 返回保存的文件路徑
    return save_cords, save_probs_aa, save_ca_probs