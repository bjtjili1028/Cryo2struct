"""
Created on 18 April 2023 1:23 AM
@author: nabin

是計算ca和胺基酸類型都有的座標位置

"""
import math
import ast
import os


# 定義一個點(Point)類別，包含座標與機率值
class Point:
    def __init__(self, x, y, z, prob=None):
        self.x = x
        self.y = y
        self.z = z
        self.prob = prob

# 計算兩點之間的歐氏距離
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# 根據給定閾值對點進行簡單聚類（貪婪聚類法）
def create_clusters(points, thres):
    clusters = []
    while points:
        cluster = [points.pop(0)]
        i = 0
        while i < len(points):
            if distance(cluster[0], points[i]) <= thres:
                cluster.append(points.pop(i))
            else:
                i += 1
        clusters.append(cluster)
    return clusters

# 從檔案讀取資料、提取座標與機率，進行聚類並計算每個聚類的質心與平均機率
def centroid_with_prob(input_file, combined_output_file, coords_output_file, prob_output_file, thres):
    points = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # 假設每行格式為 "[x, y, z], 機率值"
                coord_part, prob_part = line.split("],", 1)
                coord_str = coord_part + "]"  # 補上右括號
                coord = ast.literal_eval(coord_str)
                prob = float(prob_part.strip())
                # 只取前三個數字作為座標
                points.append(Point(coord[0], coord[1], coord[2], prob))
            except Exception as e:
                print("解析錯誤:", line, e)
                continue

    # 根據指定閾值進行聚類
    clusters = create_clusters(points, thres)

    # 分別寫入三個輸出檔案
    with open(combined_output_file, 'w') as f_combined, \
         open(coords_output_file, 'w') as f_coords, \
         open(prob_output_file, 'w') as f_prob:
        for cluster in clusters:
            n = len(cluster)
            if n == 0:
                continue
            x_avg = sum(p.x for p in cluster) / n
            y_avg = sum(p.y for p in cluster) / n
            z_avg = sum(p.z for p in cluster) / n
            prob_avg = sum(p.prob for p in cluster if p.prob is not None) / n
            f_combined.write(f"{x_avg} {y_avg} {z_avg}, {prob_avg}\n")
            f_coords.write(f"{x_avg} {y_avg} {z_avg}\n")
            f_prob.write(f"{prob_avg}\n")


# def centroid_with_prob(input_file, combined_output_file, coords_output_file, prob_output_file, thres=None):
#     """
#     不做 clustering，只負責讀取 input 檔並拆成三個輸出檔
#     """
#     with open(input_file, 'r') as f_in, \
#          open(combined_output_file, 'w') as f_combined, \
#          open(coords_output_file, 'w') as f_coords, \
#          open(prob_output_file, 'w') as f_prob:

#         for line in f_in:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 # 假設每行格式為 "[x, y, z], prob"
#                 coord_part, prob_part = line.split("],", 1)
#                 coord = ast.literal_eval(coord_part + "]")
#                 prob = float(prob_part.strip())

#                 x, y, z = coord[:3]

#                 f_combined.write(f"{x} {y} {z}, {prob}\n")
#                 f_coords.write(f"{x} {y} {z}\n")
#                 f_prob.write(f"{prob}\n")

#             except Exception as e:
#                 print("解析錯誤:", line, e)
#                 continue


# 範例使用
# if __name__ == "__main__":
#     # 例如處理 C 的資料，對 N 的資料使用相同邏輯，只需換成對應檔案名稱
#     input_filename = "34610_spilt_c_prob.txt.txt"  # 你的 C 資料檔案
#     output_filename = "cluster_centroids_c_with_prob.txt"  # 輸出結果檔案
#     threshold = 5.0  # 根據實際需求調整聚類閾值
#     centroid_with_prob(input_filename, output_filename, threshold)
    
# 主函數
def main(config_dict):
    
    # 讀取 C 和 N 的資料檔案（包含座標與機率值）
    cord_data_c = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_spilt_c_prob.txt"
    cord_data_n = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_spilt_n_prob.txt"
    
    # 設定聚類後質心與平均機率的輸出檔案路徑
    combined_output_file_c = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_combined_c.txt"
    coords_output_file_c = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_c.txt"
    save_cords_c = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_c_probs.txt"
    
    combined_output_file_n = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_combined_n.txt"
    coords_output_file_n   = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_n.txt"
    save_cords_n = f"{config_dict['input_data_dir']}/{config_dict['density_map_name']}/{config_dict['density_map_name']}_cluster_transition_n_probs.txt"
    
    # 如果輸出檔案已存在，先刪除以免影響結果
    files_to_remove = [combined_output_file_c, coords_output_file_c, save_cords_c,
                       combined_output_file_n, coords_output_file_n, save_cords_n]
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
    
    # 依據設定的閾值進行聚類，並計算每個群組的質心與平均機率
    threshold = config_dict['clustering_threshold']
    # threshold = None
    centroid_with_prob(cord_data_c, combined_output_file_c, coords_output_file_c, save_cords_c, threshold)
    centroid_with_prob(cord_data_n, combined_output_file_n, coords_output_file_n, save_cords_n, threshold)
    
    # 返回聚類結果檔案的路徑
    return (combined_output_file_c, coords_output_file_c, save_cords_c,
            combined_output_file_n, coords_output_file_n, save_cords_n)
