import numpy as np
import mrcfile
import os
import math
from copy import deepcopy
import sys

# 預期傳遞給Transformer UNet的圖像尺寸
box_size = 32  # Expected Dimensions to pass to Transformer Unet
core_size = 20  # core of the image where we dnt have to worry about boundry issues

# 創建一個清單，將原始圖像分割成一個個小區塊（大小為box_size），每個區塊會傳遞給Transformer Unet獨立處理
def create_manifest(full_image):
    # creates a list of box_size tensors. Each tensor is passed to Transformer Unet independently
    image_shape = np.shape(full_image) # 獲取圖像的形狀
    
    # 將原始圖像填充，四周填充box_size的空白區域，以避免邊界問題
    padded_image = np.zeros(
        (image_shape[0] + 2 * box_size, image_shape[1] + 2 * box_size, image_shape[2] + 2 * box_size))
    padded_image[box_size:box_size + image_shape[0], box_size:box_size + image_shape[1],
    
    # 將原始圖像放入填充後的圖像中
    box_size:box_size + image_shape[2]] = full_image
    manifest = list() # 存儲小區塊的清單

    start_point = box_size - int((box_size - core_size) / 2) # 起始點的位置，避免切割到核心區域
    cur_x = start_point # 當前的x座標
    cur_y = start_point # 當前的y座標
    cur_z = start_point # 當前的z座標
    
    # 根據core_size來遍歷整個圖像，分割成小區塊
    while cur_z + (box_size - core_size) / 2 < image_shape[2] + box_size:
        next_chunk = padded_image[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size] # 裁切出一個小區塊
        manifest.append(next_chunk) # 將小區塊加入清單中
        cur_x += core_size # 移動x座標
        if cur_x + (box_size - core_size) / 2 >= image_shape[0] + box_size:
            cur_y += core_size # 移動y坐標
            cur_x = start_point  # Reset (重置x座標)
            if cur_y + (box_size - core_size) / 2 >= image_shape[1] + box_size:
                cur_z += core_size # 移動z坐標
                cur_y = start_point  # Reset (重置y座標)
                cur_x = start_point  # Reset (重置x座標)
    return manifest # 返回所有小塊的清單

# 讀取密度圖並創建對應的蛋白質、原子和氨基酸的清單
def get_data(density_map_dir):
    protein_manifest = None
    amino_manifest = None
    atom_manifest = None
    
    # 獲取目錄中所有的檔案名稱
    processed_maps = [m for m in os.listdir(density_map_dir)]
    for maps in range(len(processed_maps)):
        os.chdir(density_map_dir)
        
        # 讀取蛋白質圖像
        if processed_maps[maps] == "emd_normalized_map.mrc":
            p_map = mrcfile.open(processed_maps[maps], mode='r')
            protein_data = deepcopy(p_map.data)  # 讀取圖像數據
            protein_manifest = create_manifest(protein_data) # 創建小區塊清單
        
        # 讀取原子圖像
        if processed_maps[maps] == "atom_emd_normalized_map.mrc":
            atom_map = mrcfile.open(processed_maps[maps], mode='r')
            atom_data = deepcopy(atom_map.data)
            atom_manifest = create_manifest(atom_data)  # 創建原子小區塊清單

        # 讀取氨基酸圖像
        if processed_maps[maps] == "amino_emd_normalized_map.mrc":
            amino_map = mrcfile.open(processed_maps[maps], mode='r')
            amino_data = deepcopy(amino_map.data)
            amino_manifest = create_manifest(amino_data) # 創建氨基酸小區塊清單

    return protein_manifest, atom_manifest, amino_manifest # 返回三個清單

# 將 Transformer Unet 的輸出重建回原始尺寸的圖像
def reconstruct_map(manifest, image_shape):
    # takes the output of Transformer Unet and reconstructs the full dimension of the protein
    extract_start = int((box_size - core_size) / 2) # 提取區塊的起始位置
    extract_end = int((box_size - core_size) / 2) + core_size # 提取區塊的結束位置
    # 計算重建後圖像的維度
    dimensions = get_manifest_dimensions(image_shape)

    reconstruct_image = np.zeros((dimensions[0], dimensions[1], dimensions[2])) # 初始化重建圖像
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
    
    # 將圖像轉為浮點型數據
    float_reconstruct_image = np.array(reconstruct_image, dtype=np.float32)
    # 裁剪至原始圖像尺寸
    float_reconstruct_image = float_reconstruct_image[:image_shape[0], :image_shape[1], :image_shape[2]]
    # 返回重建後的圖像
    return float_reconstruct_image

# 計算重建圖像的維度
def get_manifest_dimensions(image_shape):
    dimensions = [0, 0, 0]
    # 根據core_size確定最終的尺寸，確保每個維度的大小是core_size的整數倍
    dimensions[0] = math.ceil(image_shape[0] / core_size) * core_size
    dimensions[1] = math.ceil(image_shape[1] / core_size) * core_size
    dimensions[2] = math.ceil(image_shape[2] / core_size) * core_size
    return dimensions

# 創建子網格，並將每個小塊保存為壓縮的npz文件
def create_subgrids(input_data_dir, grid_division_dir):
    
    # 獲取輸入資料夾中的所有密度圖檔案名稱
    density_map_names = [m for m in os.listdir(input_data_dir)]
    for density_map_name in density_map_names:
        density_map_dir = os.path.join(input_data_dir,density_map_name)

        # 讀取數據
        protein, atom, amino = get_data(density_map_dir)
        if protein is not None and atom is not None and amino is not None:
            
            # 如果所有數據都存在，則將每個小區塊儲存到新的文件中
            for i in range(len(protein)):
                save_file_name = f'{grid_division_dir}/{density_map_name}_{i}.npz'
                np.savez_compressed(file=save_file_name, protein_grid=protein[i], atom_grid=atom[i], amino_grid=amino[i])
        else:
            print("There is no input map. Please check the input density map's directory")
            exit() # 如果缺少任何數據，則退出


if __name__ == "__main__":
    # 解析命令行參數，獲取輸入數據目錄和網格劃分目錄
    input_data_dir = sys.argv[1]
    grid_division_dir = sys.argv[2]
    os.makedirs(grid_division_dir, exist_ok=True) # 創建網格劃分目錄
    create_subgrids(input_data_dir=input_data_dir, grid_division_dir=grid_division_dir) # 創建子網格並儲存
    
