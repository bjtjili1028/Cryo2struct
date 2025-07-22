"""
@author: nabin

- Normalizes map with 95 percentile, to change the percentile value modify line number 27.

- 此腳本用於對 `.mrc` 格式的電子顯微鏡影像數據進行正規化。
- 使用 95% 分位數（percentile）來進行標準化，若要更改分位數值，可修改第 27 行。
- 正規化後的數據範圍將限制在 [0, 1] 之間。
"""
import sys
import mrcfile
from copy import deepcopy
import numpy as np
import os


def execute(inputs):
    # 記錄無法正規化的數據計數
    count = 0  
    # 獲取目錄中所有不以 .ent 結尾的文件名稱
    map_names = [fn for fn in os.listdir(inputs) if not fn.endswith(".ent")]
    
    for maps in range(len(map_names)):
        # 只處理指定的 MRC 文件
        if map_names[maps] == "emd_resampled_map.mrc":
            resample_map = map_names[maps]
            os.chdir(inputs) # 切換到輸入目錄
            print(inputs)
            
            # 讀取 MRC 文件
            clean_map = mrcfile.open(resample_map, mode='r')
            map_data = deepcopy(clean_map.data) # 複製 MRC 數據
            
            # normalize with percentile value  進行 95% 分位數正規化
            print("### Normalizing with 95-percentile for ", resample_map, " ###")
            
            try:
                percentile = np.percentile(map_data[np.nonzero(map_data)], 95)  # 計算非零元素的 95% 分位數
                map_data /= percentile # 除以分位數進行正規化
            except IndexError as error: # 若發生索引錯誤，則計數 +1
                count += 1

            # set low valued data to 0  設置數據範圍
            print("### Setting all values < 0 to 0 for ", resample_map, " ###")
            map_data[map_data < 0] = 0  # 所有小於 0 的值設為 0
            print("### Setting all values > 1 to 1 for ", resample_map, " ###")
            map_data[map_data > 1] = 1  # 所有大於 1 的值設為 1
            
            # 生成新的 MRC 文件名稱，並寫入正規化數據
            with mrcfile.new(map_names[maps].split("_")[0] + "_normalized_map.mrc", overwrite=True) as mrc:
                mrc.set_data(map_data)  # 設置數據
                mrc.voxel_size = 1 # 設置體素尺寸
                mrc.header.origin = clean_map.header.origin # 保留原始 header 的起始位置
                mrc.close()  # 關閉文件
            print("### Wrote file to ", inputs, " ###")
    print("The number of non normalized index: ", count)


if __name__ == "__main__":
    input_path = sys.argv[1]
    # 獲取所有不以 `.DS_Store` 結尾的文件名稱
    maps = [fn for fn in os.listdir(input_path) if not fn.endswith(".DS_Store")]
    for m in range(len(maps)):
         # 逐一對每個文件執行正規化
        execute(input_path +'/' + maps[m])
    print("Normalization Complete!")