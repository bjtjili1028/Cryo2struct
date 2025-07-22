import numpy as np
from scipy.ndimage import binary_dilation
import mrcfile
from scipy import ndimage as ndi



# 膨脹遮罩 - 立方體計算 (中心點周圍加多少，R=1 立方體大小3*3*3)
def _expand_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a boolean mask by `radius` voxels in 3D."""
    if radius <= 0:
        return mask
    struct = np.ones((int(radius*2+1),)*3, dtype=bool)
    return ndi.binary_dilation(mask, structure=struct)

def compute_iou(
    prediction_file: str,
    label_file: str,
    radius: int = 0,
) -> None:
    """Compute per-class and average IoU between two MRC files.

    If ``radius`` is greater than zero, each labeled voxel is expanded into a
    cubic bounding box with side ``2*radius+1`` before computing IoU.
    """
    with mrcfile.open(prediction_file, mode='r') as mrc:
        pred = mrc.data.astype(np.int64)
    with mrcfile.open(label_file, mode='r') as mrc:
        label = mrc.data.astype(np.int64)

    if pred.shape != label.shape:
        raise ValueError('Prediction and label volumes must have the same shape')

    class_ids = np.union1d(np.unique(pred), np.unique(label))
    iou_scores = {}

    for cid in class_ids:
        pred_mask = pred == cid
        label_mask = label == cid
        if radius > 0:
            pred_mask = _expand_mask(pred_mask, radius)
            label_mask = _expand_mask(label_mask, radius)
        intersection = np.logical_and(pred_mask, label_mask).sum()
        union = np.logical_or(pred_mask, label_mask).sum()
        if union == 0:
            iou_scores[cid] = float('nan')
        else:
            iou_scores[cid] = intersection / union

    for cid in sorted(iou_scores):
        val = iou_scores[cid]
        if np.isnan(val):
            print(f'class {cid}: IoU = N/A (no voxels)')
        else:
            print(f'class {cid}: IoU = {val:.4f}')

    avg_iou = np.nanmean(list(iou_scores.values()))
    print(f'Average IoU: {avg_iou:.4f}')

def compute_mask_metrics(label_vol: np.ndarray,
                         pred_vol:  np.ndarray,
                         radius:    int = 0):
    """
    Compute per-class Mask-IoU, Precision, Recall
    and overall averages between two 3D label volumes.
    Returns:
      class_metrics: dict[cid] -> {'iou', 'precision', 'recall'}
      avg_metrics:   {'iou', 'precision', 'recall'}
    """
    if label_vol.shape != pred_vol.shape:
        raise ValueError("label_vol and pred_vol must have the same shape.")
    
    class_ids = np.union1d(np.unique(label_vol),
                           np.unique(pred_vol))
    class_metrics = {}

    for cid in class_ids:
        lm = (label_vol == cid)
        pm = (pred_vol  == cid)
        if radius > 0:
            lm = _expand_mask(lm, radius)
            pm = _expand_mask(pm, radius)

        tp = np.logical_and(lm, pm).sum()
        fp = pm.sum() - tp
        fn = lm.sum() - tp
        uni = tp + fp + fn

        iou = tp/uni if uni>0 else np.nan
        precision = tp/(tp+fp) if (tp+fp)>0 else np.nan
        recall    = tp/(tp+fn) if (tp+fn)>0 else np.nan

        class_metrics[cid] = {
            'iou':       iou,
            'precision': precision,
            'recall':    recall
        }

    # 计算各指标的平均（忽略 nan）
    avg_iou       = np.nanmean([m['iou']       for m in class_metrics.values()])
    avg_precision = np.nanmean([m['precision'] for m in class_metrics.values()])
    avg_recall    = np.nanmean([m['recall']    for m in class_metrics.values()])

    avg_metrics = {
        'iou':       avg_iou,
        'precision': avg_precision,
        'recall':    avg_recall
    }

    return class_metrics, avg_metrics

# --- 讀取檔案 -----------------------------------------------------------

# 讀取透過 cryo2struct 產生的 y (label)
def read_atom_label_file(file_path):
    """
    讀 MRC label map，回傳 3D int array 以及 header_info
    """
    with mrcfile.open(file_path, mode='r') as mrc:
        data = mrc.data.copy().astype(np.int32)
        try:
            ox, oy, oz = mrc.header.origin
        except:
            ox = oy = oz = 0.0
        try:
            vx, vy, vz = mrc.voxel_size
        except:
            vx = vy = vz = 1.0

    header_info = {
        'origin':     {'x': ox,  'y': oy,  'z': oz},
        'voxel_size': {'x': vx,  'y': vy,  'z': vz},
        'shape':      data.shape
    }
    return data, header_info

# 讀取透過 cryo2struct 產生的 y (label) 後，篩選原子類別
def extract_coords(label_data, header_info, atom_type):
    """
    atom_type: 'CA', 'C', 'N' 或 'all'
    回傳 gt_pts (K×3 ndarray), gt_lbls (長度 K 的字串陣列)
    """
    origin = header_info['origin']
    vs     = header_info['voxel_size']

    # 標籤對應：1=CA, 2=N, 3=C
    label_map = {'CA':1, 'N':2, 'C':3}

    pts_list = []
    lbl_list = []

    def add_for(label_val, label_name):
        # 找所有等於這個 label 的 voxel index (z,y,x)
        idx = np.array(np.nonzero(label_data == label_val)).T
        if idx.size == 0:
            return
        # 轉成實體座標
        coords = np.zeros_like(idx, dtype=float)
        coords[:,0] = origin['x'] + idx[:,2] * vs['x']
        coords[:,1] = origin['y'] + idx[:,1] * vs['y']
        coords[:,2] = origin['z'] + idx[:,0] * vs['z']
        pts_list.append(coords)
        lbl_list.extend([label_name] * len(coords))

    # 根據 atom_type 決定要抓哪幾種
    if atom_type.lower() == 'all':
        for name, val in label_map.items():
            add_for(val, name)
    else:
        if atom_type not in label_map:
            raise ValueError("atom_type 必須是 'CA','C','N' 或 'all'")
        add_for(label_map[atom_type], atom_type)

    # 合併成單一大陣列
    if pts_list:
        gt_pts = np.vstack(pts_list)
        gt_lbls = np.array(lbl_list)
    else:
        gt_pts = np.zeros((0,3))
        gt_lbls = np.array([], dtype=str)

    return gt_pts, gt_lbls


# --- 主流程 ----------------------------------------------------------------

# 使用兩個 map 檔案進行比較

if __name__ == "__main__":
    # 1) 參數
    map_num = "26919"
    mrc_path = rf"/media/ray-suen/TRANSCEND1/huei/org_map_fasta/{map_num}/atom_emd_normalized_map.mrc"
    pred_map = rf"/media/ray-suen/TRANSCEND1/huei/pre_cluster_map/v1_split_max_cluster_3/labeled_{map_num}.mrc"
    atom_type = "all"    # 'CA', 'C', 'N' or 'all'
    print(pred_map,"\n")

    # 2) 讀 ground-truth labels & 擷取 coords
    label_data, header_info = read_atom_label_file(mrc_path)
    gt_pts, gt_lbls         = extract_coords(label_data, header_info, atom_type)
    print(f"Ground-truth {atom_type}: {len(gt_pts)} atoms")

    # 3) 讀預測 PDB（若你有多類別，就撈全部三種）
    pred_vol, _ = read_atom_label_file(pred_map)
    pr_pts, pr_lbls = extract_coords(pred_vol, header_info, atom_type)
    print(f"Predicted   {atom_type}: {len(pr_pts)} atoms")
    

    # 4) mask-level IoU - teacher
    print("\n--- mask-level IoU ---")
    compute_iou(mrc_path, pred_map, radius=1.8) # radius - 需要調整的參數

    # print("\n --- mask-level IoU2 ---")
    # class_metrics, avg_metrics = compute_mask_metrics(label_data, pred_vol, radius=1.8)

    # for cid, mets in class_metrics.items():
    #     iou_val    = mets['iou']
    #     prec_val   = mets['precision']
    #     recall_val = mets['recall']
        
    #     # 打印 IoU
    #     if np.isnan(iou_val):
    #         print(f"\nclass {cid}: \nIoU = N/A")
    #     else:
    #         print(f"\nclass {cid}: \nIoU = {iou_val:.4f}")
        
    #     # 打印 Precision
    #     if np.isnan(prec_val):
    #         print("Precision = N/A")
    #     else:
    #         print(f"Precision = {prec_val:.4f}")

    #     # 打印 Recall
    #     if np.isnan(recall_val):
    #         print("Recall = N/A")
    #     else:
    #         print(f"Recall = {recall_val:.4f}")

    # print("\n--- average metrics ---")
    # print(f"mIoU       = {avg_metrics['iou']:.4f}")
    # print(f"Precision  = {avg_metrics['precision']:.4f}")
    # print(f"Recall     = {avg_metrics['recall']:.4f}")


