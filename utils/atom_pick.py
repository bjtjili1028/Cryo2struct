"""
改進的原子選擇和分群模組，包含自適應閾值分析和智能分群
計算fasta原子數量
確認是否有重複的原子座標
"""
import ast
import mrcfile
import numpy as np
from scipy.spatial import cKDTree
from Bio.SeqIO import parse as fasta_parse
# from clustering_centroid import Point # change import path as needed
import re
###############################################
import math
from collections import deque
from typing import List, Tuple

# --- 資料結構定義 ---

class Point:
    """一個代表三維空間點的簡單類別"""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

###############################################

# def determine_target_count(fasta_file):
#     """估計目標原子數"""
#     print("\n=== 目標原子數量估計 ===")
    
#     total_len = 0
#     with open(fasta_file, "r") as handle:
#         for record in fasta_parse(handle, "fasta"):
#             chain_id = record.id.split('|')[1].replace('Chain ', '') if '|' in record.id else record.id
#             print(f"  鏈 {chain_id}: {len(record.seq)} 個氨基酸")
#             total_len += len(record.seq)
#     print(f"  總計: {total_len} 個氨基酸")
    
#     return total_len


import re

def determine_target_count(fasta_file):
    print("\n=== 目標序列總長度估計（含多鏈） ===")
    total_len = 0

    with open(fasta_file, "r") as handle:
        for record in fasta_parse(handle, "fasta"):
            seq_len = len(record.seq)
            desc = record.description

            # 預設 1 條 chain
            n_chain = 1

            # 抓 Chain(s) ... 到下一個 |
            m = re.search(r"Chain[s]?\s+([^|]+)", desc)
            if m:
                chain_block = m.group(1)
                # 用逗號切，數有幾段
                chains = [c.strip() for c in chain_block.split(",")]
                n_chain = len(chains)

            print(f"  {record.id}: {n_chain} 條鏈 × {seq_len}")

            total_len += n_chain * seq_len

    print(f"  總計: {total_len}")
    return total_len



# --- 資料結構定義 ---
class WeightedPoint(Point):
    """帶有機率權重的點，用於分群和質心計算。"""
    def __init__(self, x: float, y: float, z: float, prob: float):
        super().__init__(x, y, z)
        self.prob = prob

# --- 新增：自適應閾值分析 ---
def adaptive_threshold_analysis(prob_file, target_count):
    """
    更智能的閾值分析，針對不同原子類型找到最優閾值
    """
    print(f"\n=== 自適應閾值分析 (目標: {target_count} 個原子) ===")
    
    ca_probs, n_probs, c_probs = [], [], []
    with open(prob_file, "r") as f:
        for line in f:
            try:
                vals = ast.literal_eval(line) # 把一整行轉成 Python 物件：vals = [[x,y,z], p0, p1, p2, p3]
                if len(vals) == 5:
                    ca_probs.append(vals[2])
                    n_probs.append(vals[3])
                    c_probs.append(vals[4])
            except Exception:
                continue
    
    ca_probs.sort(reverse=True) # 降序排序
    n_probs.sort(reverse=True)
    c_probs.sort(reverse=True)
    
    # 動態找到接近目標數量的閾值
    def find_optimal_threshold(probs, target):
        if len(probs) < target:
            return 0.1  # 如果樣本不足，使用最低閾值，結束執行
        
        # 取第target個元素作為閾值，確保至少有target個點
        optimal_prob = probs[min(target-1, len(probs)-1)] 
        # 稍微降低閾值以增加候選點
        return max(0.1, optimal_prob * 0.95)
    
    ca_optimal = find_optimal_threshold(ca_probs, target_count)
    n_optimal = find_optimal_threshold(n_probs, target_count)
    c_optimal = find_optimal_threshold(c_probs, target_count)
    
    print(f"智能閾值建議:")
    print(f"  CA: {ca_optimal:.3f} (預期 ~{sum(1 for p in ca_probs if p >= ca_optimal)} 個點)")
    print(f"  N:  {n_optimal:.3f} (預期 ~{sum(1 for p in n_probs if p >= n_optimal)} 個點)")
    print(f"  C:  {c_optimal:.3f} (預期 ~{sum(1 for p in c_probs if p >= c_optimal)} 個點)")
    
    return ca_optimal, n_optimal, c_optimal

def parse_probabilities(prob_file, ca_threshold, n_threshold, c_threshold):
    """根據各原子類型的閾值過濾點"""
    ca_points, n_points, c_points = [], [], []
    with open(prob_file, "r") as f:
        for line in f:
            try:
                vals = ast.literal_eval(line) # 把一整行轉成 Python 物件：vals = [[x,y,z], p0, p1, p2, p3]
                if len(vals) != 5: continue
                x, y, z = vals[0]
                _, ca_p, n_p, c_p = vals[1:]
                
                if ca_p >= ca_threshold:
                    ca_points.append(WeightedPoint(x, y, z, ca_p))
                if n_p >= n_threshold:
                    n_points.append(WeightedPoint(x, y, z, n_p))
                if c_p >= c_threshold:
                    c_points.append(WeightedPoint(x, y, z, c_p))
            except Exception:
                continue
    return ca_points, n_points, c_points # 回傳 [x,y,z,prob] 格式的點列表

def nms_kdtree_adaptive(points, radius, max_points=None):
    """
    自適應NMS（Non-Maximum Suppression）：根據輸出點數動態調整半徑，以控制最終保留的點數
    points: 輸入的點列表，每個點需有屬性 x, y, z, prob
    radius: 初始抑制半徑
    max_points: 最多保留的點數（可選）
    """
    if not points:
        return []
    
    original_radius = radius # 初次輸入的半徑
    print(f"初始半徑為：{original_radius}")
    current_radius = radius # 當前半徑 (會根據迴圈被替代)
    
    iteration = 0  # 迭代次數
    max_iterations = 5 # 最大迭代次數
    
    # 迭代調整半徑，直到滿足條件或超過最大迭代次數
    while iteration < max_iterations:
        
        coords = np.array([[p.x, p.y, p.z] for p in points]) # 將點的座標轉為numpy陣列
        
        probs = np.array([p.prob for p in points]) # 提取機率值
        sorted_indices = np.argsort(-probs, kind='stable') # 建立機率從高到低的索引排序
        
        # 建立KD樹以加速鄰居查詢
        tree = cKDTree(coords)
        
        suppressed_indices = set() # 被抑制的點索引集合
        kept_indices = [] # 最終保留的點索引
        
        # 按機率從高到低逐一檢查
        for i in sorted_indices:
            if i in suppressed_indices: 
                continue # 已被抑制，跳過
            
            kept_indices.append(i) # 保留此點
            
            # 查詢半徑內的鄰居點索引
            neighbors = tree.query_ball_point(coords[i], r=current_radius)
            
            # 若鄰居點索引中有其他點，則將它們標記為被抑制
            for neighbor_idx in neighbors:
                if neighbor_idx != i:
                    suppressed_indices.add(neighbor_idx) # 抑制鄰居點
        
        result_count = len(kept_indices) # 此輪保留的點數
        
         # 根據max_points限制，自適應調整半徑
        if max_points and result_count > max_points * 1.5:
            current_radius *= 1.2  # 增加半徑以減少點數
            iteration += 1
            continue
        elif max_points and result_count < max_points * 0.7:
            current_radius *= 0.8  # 減少半徑以增加點數
            iteration += 1
            continue
        else:
            break
    
    if current_radius != original_radius:
        print(f"    自適應NMS: 半徑調整 {original_radius:.2f} → {current_radius:.2f}, 得到 {result_count} 個點")
    
    return [points[i] for i in kept_indices],current_radius


# 確認是否有相同的座標位置
def _as_xyzp_any(it):
    if hasattr(it, "x") and hasattr(it, "y") and hasattr(it, "z"):
        return float(it.x), float(it.y), float(it.z), float(getattr(it, "prob", 1.0))
    x, y, z = float(it[0]), float(it[1]), float(it[2])
    p = float(it[3]) if len(it) >= 4 else 1.0
    return x, y, z, p

def _voxelize(items, geom_mrc_path):
    """把連續座標映到 geom_mrc 的 voxel；回傳 (voxels_set, unique_count, duplicate_count)。"""
    if not items:
        return set(), 0, 0
    with mrcfile.open(geom_mrc_path, mode='r') as m:
        Z, Y, X = m.data.shape
        try:
            ox, oy, oz = float(m.header.origin.x), float(m.header.origin.y), float(m.header.origin.z)
        except Exception:
            o = m.header.origin; ox, oy, oz = float(o[0]), float(o[1]), float(o[2])
        try:
            vx, vy, vz = float(m.voxel_size.x), float(m.voxel_size.y), float(m.voxel_size.z)
        except Exception:
            vs = m.voxel_size; vx, vy, vz = float(vs[0]), float(vs[1]), float(vs[2])

    vox = []
    for it in items:
        x, y, z, _ = _as_xyzp_any(it)
        ix = int(round((x - ox) / vx))
        iy = int(round((y - oy) / vy))
        iz = int(round((z - oz) / vz))
        if 0 <= iz < Z and 0 <= iy < Y and 0 <= ix < X:
            vox.append((iz, iy, ix))
    vox_set = set(vox)
    unique_cnt = len(vox_set)
    dup_cnt = max(0, len(vox) - unique_cnt)
    return vox_set, unique_cnt, dup_cnt

def compute_overlap_stats(ca_items, n_items, c_items, geom_mrc_path, stage_tag):
    """
    回傳並印出：跨類別重疊（CA&N、CA&C、N&C、三者同位）以及同類別內的重複 voxel 數。
    """
    S_ca, U_ca, D_ca = _voxelize(ca_items, geom_mrc_path)
    S_n,  U_n,  D_n  = _voxelize(n_items,  geom_mrc_path)
    S_c,  U_c,  D_c  = _voxelize(c_items,  geom_mrc_path)

    ca_n = len(S_ca & S_n)
    ca_c = len(S_ca & S_c)
    n_c  = len(S_n  & S_c)
    all3 = len(S_ca & S_n & S_c)

    exist_cross = (ca_n > 0) or (ca_c > 0) or (n_c > 0)

    print(f"\n[overlap@{stage_tag}] cross_exist={exist_cross} | pairs CA&N={ca_n}, CA&C={ca_c}, N&C={n_c}, triple={all3} | "
          f"same-class dups: CA={D_ca}, N={D_n}, C={D_c}")

    return {
        "pairs": {"CA&N": ca_n, "CA&C": ca_c, "N&C": n_c},
        "triple": all3,
        "cross_exist": exist_cross,
        "same_class_dups": {"CA": D_ca, "N": D_n, "C": D_c},
        "unique_voxels": {"CA": U_ca, "N": U_n, "C": U_c}
    }

#############################################################################################


def resolve_cross_class_overlaps_keep_maxprob(ca_items, n_items, c_items, geom_mrc_path, tie_priority=("CA","N","C")):
    """
    將 CA / N / C 三類中『落在同一體素(以 geom_mrc 幾何離散)』的點，做跨類去重：
    - 只保留該體素中「prob」最大的那一個
    - 若剛好同分，依 tie_priority（預設 CA > N > C）保留
    回傳：去重後的 (ca_items2, n_items2, c_items2)
    並印出 before/after 與刪除統計。
    """
    import numpy as _np
    import mrcfile as _mrc

    def _as_xyzp(it):
        # 支援 (x,y,z,p) tuple 或有 .x .y .z (.prob) 的物件
        if hasattr(it, "x") and hasattr(it, "y") and hasattr(it, "z"):
            return float(it.x), float(it.y), float(it.z), float(getattr(it, "prob", 1.0))
        x, y, z = float(it[0]), float(it[1]), float(it[2])
        p = float(it[3]) if len(it) >= 4 else 1.0
        return x, y, z, p

    # 讀幾何（用你評估 IoU / 寫 MRC 用的同一個 grid）
    with _mrc.open(geom_mrc_path, mode='r') as m:
        shape = m.data.shape
        # origin
        try:
            ox, oy, oz = float(m.header.origin.x), float(m.header.origin.y), float(m.header.origin.z)
        except Exception:
            o = m.header.origin; ox, oy, oz = float(o[0]), float(o[1]), float(o[2])
        # voxel
        try:
            vx, vy, vz = float(m.voxel_size.x), float(m.voxel_size.y), float(m.voxel_size.z)
        except Exception:
            vs = m.voxel_size; vx, vy, vz = float(vs[0]), float(vs[1]), float(vs[2])

    tie_rank = {name: i for i, name in enumerate(tie_priority)}  # 小的優先

    # 將三類點丟到同一個 dict：key=voxel(iz,iy,ix)，val=(cls_name, idx_in_list, prob, raw_item)
    vox_best = {}   # key -> {'cls':"CA"/"N"/"C", 'prob':p, 'rank':tie_rank, 'item':it}
    removed = {"CA":0, "N":0, "C":0}

    def _voxel_of(x, y, z):
        ix = int(round((x - ox) / vx))
        iy = int(round((y - oy) / vy))
        iz = int(round((z - oz) / vz))
        return iz, iy, ix

    def _feed(items, cls_name):
        nonlocal vox_best, removed
        Z, Y, X = shape
        for it in items:
            x, y, z, p = _as_xyzp(it)
            iz, iy, ix = _voxel_of(x, y, z)
            if not (0 <= iz < Z and 0 <= iy < Y and 0 <= ix < X):
                # 超出體素範圍，保留原樣（不做格內去重）
                key = None
            else:
                key = (iz, iy, ix)

            if key is None:
                # 以獨立 key 儲存，避免被誤去重
                key = ("_oo_", id(it))  # out-of-grid guard key

            if key not in vox_best:
                vox_best[key] = {"cls": cls_name, "prob": p, "rank": tie_rank.get(cls_name, 999), "item": it}
            else:
                cur = vox_best[key]
                # 只在「同一 voxel」時比較（_oo_ key 不會重複）
                replace = False
                if p > cur["prob"]:
                    replace = True
                elif p == cur["prob"] and tie_rank.get(cls_name, 999) < cur["rank"]:
                    replace = True
                if replace:
                    # 被替換掉的那個算 removed
                    removed[cur["cls"]] += 1
                    vox_best[key] = {"cls": cls_name, "prob": p, "rank": tie_rank.get(cls_name, 999), "item": it}
                else:
                    removed[cls_name] += 1

    _feed(ca_items, "CA")
    _feed(n_items,  "N")
    _feed(c_items,  "C")

    # 收集結果
    out_ca, out_n, out_c = [], [], []
    for v in vox_best.values():
        if v["cls"] == "CA": out_ca.append(v["item"])
        elif v["cls"] == "N": out_n.append(v["item"])
        else: out_c.append(v["item"])

    print(f"[cross-resolve@final] before CA/N/C = {len(ca_items)}/{len(n_items)}/{len(c_items)} "
          f"| removed CA/N/C = {removed['CA']}/{removed['N']}/{removed['C']} "
          f"| after CA/N/C = {len(out_ca)}/{len(out_n)}/{len(out_c)} "
          f"| tie_policy = prob_max → {tie_priority}")
    return out_ca, out_n, out_c

##############################################################################3
# 要輸出資料的點讓後面可以用
def write_centroid_file(centroids, out_path):
    """將質心寫入文本文件（支援 tuple/list 或 WeightedPoint 物件）"""
    if not out_path:
        return

    with open(out_path, 'w') as f:
        for item in centroids:
            # 情況 1：item 是 (x,y,z,prob) 這種可解包資料
            if isinstance(item, (tuple, list)) and len(item) >= 4:
                x, y, z, prob = item[:4]
            else:
                # 情況 2：item 是 WeightedPoint / 類似物件
                x = getattr(item, "x")
                y = getattr(item, "y")
                z = getattr(item, "z")

                # prob 欄位命名可能不同：prob / weight / w / score
                prob = getattr(item, "prob", None)
                if prob is None:
                    prob = getattr(item, "weight", None)
                if prob is None:
                    prob = getattr(item, "w", None)
                if prob is None:
                    prob = getattr(item, "score", None)

                if prob is None:
                    raise TypeError(
                        f"Centroid object has no prob/weight field: {type(item)}"
                    )

            f.write(f'[{x:.3f}, {y:.3f}, {z:.3f}], {prob:.4f}\n')

    print(f"寫入 {len(centroids)} 個質心到 {out_path}")
