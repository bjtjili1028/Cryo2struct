"""
@author: nabin
本腳本在無 GUI 模式下運行 ChimeraX 來對 `.map` 格式的電子顯微鏡影像數據進行重採樣（resampling）。
- Runs ChimeraX in no-GUI mode to resample map.
"""
import sys
import subprocess
import os

# ------------------------- linux下執行 -----------------------------
# def execute(input_path, chimera_path):
#     """
#     creates resampling script and executes them in Chimera
#     :return:
    
#     創建 ChimeraX 腳本並執行它們以進行重採樣。
#     :param input_path: 包含 .map 文件的輸入目錄
#     :param chimera_path: ChimeraX 可執行文件的路徑
#     """
    
#     # 獲取目錄中所有不以 .ent 和 .DS_Store 結尾的文件名稱
#     map_names = [fn for fn in os.listdir(input_path) if not fn.endswith(".ent") if not fn.endswith(".DS_Store")]
#     if map_names is None:
#         print("### Please check the directory!! No input files present in", input_path, '###')
#         exit()
#     for maps in range(len(map_names)):
#         path = os.path.join(input_path, map_names[maps]) # 獲取完整文件路徑
#         emd_map = [e for e in os.listdir(path) if e.endswith(".map")] # 獲取所有 .map 文件
#         for density_maps in range(len(emd_map)):
#             print(f"Working on : {emd_map[density_maps]}")
#             # 創建 ChimeraX 腳本文件
#             chimera_scripts = open('resample.cxc', 'w')
#             chimera_scripts.write('open ' + path + '/' + emd_map[density_maps] + '\n'
#                                   'vol resample #1 spacing 1.0 \n' # 以體素大小 1.0 進行重採樣
#                                   'save ' + path + '/' + emd_map[density_maps].split("_")[0] + '_resampled_map.mrc' + ' model #2 \n'
#                                   'exit') # 儲存為新的 MRC 文件，然後退出 ChimeraX

#             chimera_scripts.close()
#             script_finished = False # 記錄腳本是否成功執行
#             while not script_finished:
#                 try:
#                     # 執行 ChimeraX，並運行剛剛生成的 resampling 腳本
#                     subprocess.run([chimera_path, '--nogui', chimera_scripts.name])
#                     script_finished = True # 如果執行成功則結束迴圈
#                 except FileNotFoundError as error:
#                     raise error # 如果找不到 ChimeraX 可執行文件，則拋出錯誤
#             # 刪除臨時腳本文件
#             os.remove(chimera_scripts.name)
#             print(f'### Resampled {emd_map[density_maps]} and saved on new grid with voxel size of {1} ###')
# ------------------------------------------------------




#  將 WSL 格式的 `/mnt/...` 路徑轉換成 Windows `C:\...` 格式
def convert_to_windows_path(wsl_path):
    result = subprocess.run(['wslpath', '-w', wsl_path], capture_output=True, text=True)
    return result.stdout.strip()

# ------------------------- windows下執行 -----------------------------

def execute(input_path, chimera_path):
    """
    Creates resampling script and executes it in ChimeraX
    """
    # 獲取輸入資料夾中的所有檔案（排除 `.ent` 和 `.DS_Store`）
    map_names = [fn for fn in os.listdir(input_path) if not fn.endswith(".ent") and not fn.endswith(".DS_Store")]

    # 如果目錄內沒有可處理的資料夾，則顯示錯誤訊息並結束程式
    if map_names is None:
        print("### Please check the directory!! No input files present in", input_path, '###')
        exit()

    # 逐一處理每個資料夾
    for maps in range(len(map_names)):
        path = os.path.join(input_path, map_names[maps]) # 取得完整的資料夾路徑
        emd_map = [e for e in os.listdir(path) if e.endswith(".map")]  # 找出所有 `.map` 檔案

        # 逐一處理每個 `.map` 檔案
        for density_maps in range(len(emd_map)):
            
            # 文件路徑 (for windows) 
            full_path = os.path.join(path, emd_map[density_maps])

            # 轉換路徑 (for windows)
            win_path = convert_to_windows_path(full_path)

            # 產生 ChimeraX 指令腳本
            chimera_scripts = open('resample.cxc', 'w')
            
            # 建立 ChimeraX 指令腳本 `.cxc`
            chimera_scripts.write(f'open "{win_path}"\n'  # 開啟 .map 檔案 (for windows)
                                  'vol resample #1 spacing 1.0 \n'  # 進行重取樣，設定 voxel size 為 1.0
                                  f'save "{win_path.split(".")[0]}_resampled.mrc" model #2 \n'  # 存放 .map 檔案 (for windows)
                                  'exit') # 儲存為 .mrc            
            
            chimera_scripts.close()
            
            # 標記腳本是否成功執行
            script_finished = False
           
            while not script_finished:
                try:
                    # 執行 ChimeraX 並載入腳本
                    subprocess.run([chimera_path, '--nogui', chimera_scripts.name])
                    script_finished = True
                except FileNotFoundError as error:
                    raise error # 如果找不到 ChimeraX，拋出錯誤
            
            # 刪除執行過的腳本文件
            os.remove(chimera_scripts.name)
            print(f'### Resampled {full_path} and saved on new grid with voxel size of {1} ###')      


if __name__ == "__main__":
    # 獲取命令列參數中的輸入目錄
    input_path = sys.argv[1]
    # output_path = sys.argv[2]

    # 設定 ChimeraX 可執行文件的路徑，若未提供則使用預設路徑
    if len(sys.argv) > 2:
        chimera_path = sys.argv[2]
    else:
        # chimera_path = '/usr/bin/chimerax' # 預設為 Linux 環境的 ChimeraX 路徑
        chimera_path = "/mnt/d/ChimeraX 1.4/bin/ChimeraX-console.exe"
        
    # 執行重採樣過程
    execute(input_path, chimera_path)
    print("Resampling Complete!")
