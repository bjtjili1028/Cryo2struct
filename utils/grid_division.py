import numpy as np  # 引入數值處理庫
import mrcfile  # 用於讀取 MRC 格式的文件
import os  # 用於處理目錄和文件
import math  # 用於數學運算（例如取整和上限運算）
from copy import deepcopy  # 用於深拷貝對象

box_size = 32  # Expected Dimensions to pass to Transformer Unet (定義每個小塊的尺寸，將作為Transformer Unet模型的輸入尺寸)
core_size = 20  # core of the image where we dnt have to worry about boundry issues (圖像的核心區域大小，處理邊界問題時不需要擔心此區域)

# 創建清單，將大圖像切割成小塊，每個小塊傳遞給Transformer Unet進行處理
def create_manifest(full_image):
    # creates a list of box_size tensors. Each tensor is passed to Transformer Unet independently
    
    image_shape = np.shape(full_image) # 獲取圖像的尺寸
    
    # 在原圖像周圍添加填充，確保處理過程中邊界不會引發問題
    padded_image = np.zeros(
        (image_shape[0] + 2 * box_size, image_shape[1] + 2 * box_size, image_shape[2] + 2 * box_size))
    
    # 將原始圖像放入填充後的圖像中 (從第32放到原始圖像+32，確定旁邊一定有留32的邊界)
    padded_image[box_size:box_size + image_shape[0], box_size:box_size + image_shape[1], box_size:box_size + image_shape[2]] = full_image 
    
    # 用於存儲每個小塊的清單
    manifest = list()

    # 計算切割起始位置，保證核心區域不受邊界影響
    start_point = box_size - int((box_size - core_size) / 2) 
    
    # 當前 X、Y、Z 座標
    cur_x = start_point
    cur_y = start_point
    cur_z = start_point
 
    # 遍歷整個圖像，按照 core_size 進行切割，生成小塊
    while cur_z + (box_size - core_size) / 2 < image_shape[2] + box_size:
        
        # 將每個小塊添加到manifest中
        next_chunk = padded_image[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        manifest.append(next_chunk) 
        
        # 移動x坐標
        cur_x += core_size
        
        if cur_x + (box_size - core_size) / 2 >= image_shape[0] + box_size: # 超過要換行
            cur_y += core_size # 移動y坐標
            cur_x = start_point  # Reset (重置x座標)
            
            if cur_y + (box_size - core_size) / 2 >= image_shape[1] + box_size: # 超過要換列
                cur_z += core_size # 移動z座標
                cur_y = start_point  # Reset (重置y座標)
                cur_x = start_point  # Reset (重置x座標)
    return manifest # 返回所有小塊的清單


# 從密度圖目錄中讀取處理後的圖像數據
def get_data(density_map_dir):
    # 獲取目錄中的所有檔案
    processed_maps = [m for m in os.listdir(density_map_dir)]
    
    # 根據目錄切換當前資料夾
    for maps in range(len(processed_maps)):
        os.chdir(density_map_dir)
    
        # 查找特定的MRC文件
        if processed_maps[maps] == "emd_normalized_map.mrc":
            p_map = mrcfile.open(processed_maps[maps], mode='r') # 打開MRC檔案
            protein_data = deepcopy(p_map.data) # 複製數據
            protein_manifest = create_manifest(protein_data) # 創建子網格清單
    
    return protein_manifest # 返回創建的子網格清單


# 根據Transformer Unet的輸出重建完整的蛋白質密度圖
def reconstruct_map(manifest, image_shape):
    # takes the output of Transformer Unet and reconstructs the full dimension of the protein
    extract_start = int((box_size - core_size) / 2) # 計算提取區域的起始位置
    extract_end = int((box_size - core_size) / 2) + core_size # 計算提取區域的結束位置
    dimensions = get_manifest_dimensions(image_shape) # 獲取重建圖像的尺寸

    reconstruct_image = np.zeros((dimensions[0], dimensions[1], dimensions[2])) # 創建一個空的重建圖像
    counter = 0 # 計數器，用於遍歷 manifest(小格清單)
    for z_steps in range(int(dimensions[2] / core_size)):
        for y_steps in range(int(dimensions[1] / core_size)):
            for x_steps in range(int(dimensions[0] / core_size)):
                # 將每個小區塊放回到原始圖像的位置
                reconstruct_image[x_steps * core_size:(x_steps + 1) * core_size,
                y_steps * core_size:(y_steps + 1) * core_size, z_steps * core_size:(z_steps + 1) * core_size] = \
                    manifest[counter][extract_start:extract_end, extract_start:extract_end,
                    extract_start:extract_end]
                counter += 1 # 更新計數器
                
    float_reconstruct_image = np.array(reconstruct_image, dtype=np.float32) # 將重建圖像轉換為float32格式
    float_reconstruct_image = float_reconstruct_image[:image_shape[0], :image_shape[1], :image_shape[2]]  # 裁剪至原始圖像尺寸
    return float_reconstruct_image # 返回重建的圖像

# 根據原始圖像尺寸計算manifest的維度
def get_manifest_dimensions(image_shape):
    dimensions = [0, 0, 0]
    # 根據core_size確定最終的尺寸，確保每個維度的大小是core_size的整數倍
    dimensions[0] = math.ceil(image_shape[0] / core_size) * core_size
    dimensions[1] = math.ceil(image_shape[1] / core_size) * core_size
    dimensions[2] = math.ceil(image_shape[2] / core_size) * core_size
    return dimensions # 返回計算後的維度

# 創建子網格，並將每個小塊保存為壓縮的npz文件
def create_subgrids(input_data_dir, density_map_name):
    # 獲取密度圖目錄的路徑
    density_map_dir = os.path.join(input_data_dir,density_map_name)
    # 獲取分割後的子網格
    protein = get_data(density_map_dir)
    if protein is not None:
        # 創建保存子網格的目錄
        split_map_dir = os.path.join(density_map_dir, f"{density_map_name}_splits")
        os.makedirs(split_map_dir, exist_ok=True)
        # 將每個子網格保存為壓縮的npz文件
        for i in range(len(protein)):
            save_file_name = f'{split_map_dir}/{density_map_name}_{i}.npz'
            np.savez_compressed(file=save_file_name, protein_grid=protein[i])
    else:
        # 如果未找到輸入圖像，則提示錯誤
        print("There is no input map. Please check the input density map's directory")
        exit()
