import argparse  # 用於解析命令列參數
import os       # 用於檔案路徑處理
import math      # 提供數學函式，如 floor
import mrcfile   # 用於讀取與寫入 MRC 格式檔案
import numpy as np  # 數值計算核心套件


def get_index(cord: float, origin: float, voxel: float) -> int:
    """
    將真實座標轉換為 MRC 陣列索引
    cord: 座標值；origin: 原點偏移；voxel: 每格尺寸
    回傳對應的整數索引
    """
    return int(math.floor((cord - origin) / voxel))


def load_centroids_single_folder(input_dir: str, map_name: str):
    """
    從同一個資料夾中讀取單一 map 的 CA/N/C 座標與機率檔，
    檔名格式：
      {map_name}_CA_coord.txt, {map_name}_CA_prob.txt
      {map_name}_N_coord.txt,  {map_name}_N_prob.txt
      {map_name}_C_coord.txt,  {map_name}_C_prob.txt
    回傳三個 (x,y,z,p) 清單：ca_pts, n_pts, c_pts
    """
    def load_pair(atom: str):
        coords_path = os.path.join(input_dir, f"{map_name}_cluster_transition_{atom}.txt")
        prob_path= os.path.join(input_dir, f"{map_name}_cluster_transition_{atom}_probs.txt")
        pts = []
        # 讀取座標
        with open(coords_path, 'r') as fc:
            coords = [tuple(map(float, line.split())) for line in fc]
        # 讀取機率
        with open(prob_path, 'r') as fp:
            probs = [float(line.strip()) for line in fp]
        # 合併為 (x,y,z,p)
        for (x, y, z), p in zip(coords, probs):
            pts.append((x, y, z, p))
        return pts

    ca_pts = load_pair('CA')
    n_pts  = load_pair('N')
    c_pts  = load_pair('C')
    return ca_pts, n_pts, c_pts


def write_mrc(ca_pts, n_pts, c_pts, ref_map_path: str, out_path: str):
    """
    使用參考 MRC 檔的結構與 metadata，
    把 CA/N/C 質心標記為 1/2/3，並輸出新的 MRC 檔
    """
    with mrcfile.open(ref_map_path, mode='r') as m_ref:
        data = np.zeros_like(m_ref.data, dtype=np.int16)
        origin = m_ref.header.origin
        x0, y0, z0 = origin['x'], origin['y'], origin['z']
        vx, vy, vz = (
            m_ref.voxel_size['x'],
            m_ref.voxel_size['y'],
            m_ref.voxel_size['z'],
        )

        def place(points, label):
            """
            把質心放入對應體素並標記數值；
            points: (x,y,z,p) 列表；label: 整數標籤
            """
            for x, y, z, _ in points:
                iz = get_index(z, z0, vz)
                jy = get_index(y, y0, vy)
                ix = get_index(x, x0, vx)
                if (
                    0 <= iz < data.shape[0] and
                    0 <= jy < data.shape[1] and
                    0 <= ix < data.shape[2]
                ):
                    data[iz, jy, ix] = label

        place(ca_pts, 1)
        place(n_pts, 2)
        place(c_pts, 3)

    with mrcfile.new(out_path, overwrite=True) as m_out:
        m_out.set_data(data.astype(np.float32))  # MRC 資料需為 float32
        m_out.voxel_size = vx
        m_out.header.origin = origin


def main():
    parser = argparse.ArgumentParser(
        description='從同一資料夾讀取單一 map 的三原子 txt，並合併生成標籤 MRC'
    )
    # 使用 -- 參數代替位置引數
    parser.add_argument('--input_dir', required=True,
                        help='含有 txt 檔與 map 檔的資料夾路徑')
    parser.add_argument('--map_name',  required=True,
                        help='地圖名稱（對應 txt 檔前綴及 .mrc 檔名）')
    parser.add_argument('--output',    required=True,
                        help='輸出的標籤 MRC 檔路徑')
    args = parser.parse_args()

    # 自動組裝參考 MRC 路徑
    ref_map_path = os.path.join(args.input_dir, "emd_normalized_map.mrc")

    # 從同一資料夾載入該 map 的 CA/N/C 質心資料
    ca_pts, n_pts, c_pts = load_centroids_single_folder(
        args.input_dir, args.map_name
    )

    # 生成標籤 MRC
    write_mrc(ca_pts, n_pts, c_pts, ref_map_path, args.output)

if __name__ == '__main__':
    main()

# python output_cluster_map.py --input_dir /media/ray-suen/TRANSCEND1/huei/Cryo2struct/input/11978  --map_name 11978  --output labeled_11978.mrc 