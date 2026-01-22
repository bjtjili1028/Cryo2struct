# ------------------------------------------------------------
# 目的：在三類點雲（CA、N、C）中，為每個 CA 儘量配到一個 C（必要），
#      再視情況配到一個 N（可省略），以距離 +（可選）角度成本作為評分。
# 流程：Greedy + Two-Step（先 CA↔C，再 CA↔N）
# ------------------------------------------------------------

from dataclasses import dataclass, replace
from typing import List, Tuple, Optional, Iterable
import math
import numpy as np
from scipy.spatial import cKDTree

# =========================
# 可調參（常用）— 你最常改這裡
# =========================
@dataclass
class Params:
    # ---- 半徑（搜尋上限，越小越嚴格）----
    r_ca_c_max: float = 2.4   # CA 搜 C 的最大半徑（Å）
    r_ca_n_max: float = 2.4   # CA 搜 N 的最大半徑（Å）

    r_ca_c_low: float = 1.0
    r_ca_n_low: float = 1.0

    # ---- 是否啟用角度條件 ----
    use_angle: bool = False
    angle_target: float = 110.0   # 期望的 ∠N–CA–C（度）
    angle_tol: float = 25.0       # 允許偏離（度）；越小越嚴格
    w_angle: float = 0.7          # 角度成本權重；越大角度越重要

    # ---- 鍵長視窗（開啟會更嚴格）----
    bondlen_window: bool = False  # 是否啟用鍵長視窗
    ca_c_len_lo: float = 1.3
    ca_c_len_hi: float = 1.9
    ca_n_len_lo: float = 1.2
    ca_n_len_hi: float = 1.8

    # ---- 距離成本設定 ----
    distance_power: float = 1.5   # 距離成本的次方；1~2 常見
    w_dist_c: float = 1.2         # C 距離權重
    w_dist_n: float = 1.2         # N 距離權重

    # ---- 匹配策略 ----
    exclusive_match: bool = True  # True：同一顆 C/N 不會被多個 CA 重複使用
    
    # ---- 不啟用角度時，使用距離為先還是機率為先  ----
    """
    候選排序優先權：
      - "distance_then_prob": 先依距離小→大，再依 prob 高→低 # greed 1
      - "prob_then_distance": 先依 prob 高→低，再依距離小→大 # greed 2
    """
    select_priority: str = "prob_then_distance" 

# =========================
# 輸入點資料結構
# =========================
@dataclass
class WPoint:
    x: float
    y: float
    z: float
    prob: float = 1.0  # 置信度（高者先配）

    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def as_tuple(self):
        return (self.x, self.y, self.z, self.prob)


def to_wpoints(points: Iterable) -> List[WPoint]:
    """
    將輸入格式（WPoint / 具 x,y,z 的物件 / (x,y,z[,prob])）統一轉為 WPoint。
    """
    out: List[WPoint] = []
    for p in points:
        if isinstance(p, WPoint):
            out.append(p)
        elif hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
            out.append(WPoint(float(p.x), float(p.y), float(p.z), float(getattr(p, "prob", 1.0))))
        else:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            prob = float(p[3]) if len(p) >= 4 else 1.0
            out.append(WPoint(x, y, z, prob))
    return out


# =========================
# 幾何工具
# =========================
def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def angle_n_ca_c(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> float:
    """
    以 CA 為頂點計算 ∠N–CA–C（度）
    """
    v1 = n - ca
    v2 = c - ca
    denom = max(1e-12, np.linalg.norm(v1) * np.linalg.norm(v2))  # 防 0
    cosv = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosv))

def angle_penalty(theta: float, target: float, tol: float) -> float:
    """
    角度懲罰（越接近 target 越小）。這裡用 (|θ-目標| / tol) 作為懲罰（容忍度內 ~<=1）。
    """
    return abs(theta - target) / max(1e-6, tol)

def in_window(d: float, lo: float, hi: float, use_window: bool) -> bool:
    """
    若有啟用鍵長視窗，距離必須落在 [lo, hi]；否則一律 True。
    """
    return (lo <= d <= hi) if use_window else True


# =========================
# Greedy + Two-Step 主邏輯
# =========================
def greedy_two_step(
    ca_pts: List[WPoint],
    c_pts: List[WPoint],
    n_pts: List[WPoint],
    p: Params
) -> Tuple[List[WPoint], List[WPoint], List[WPoint]]:
    """
    依 CA 的 prob 由高到低：
      1) 為 CA 找一顆最佳 C（距離成本）。
      2) 嘗試為 CA 找一顆最佳 N（若無符合條件 → 留空）。
    備註：若 exclusive_match=True，已被使用的 C/N 不會再次分配。
    """
    if not ca_pts or not c_pts:
        return [], [], []

    ca_xyz = np.stack([w.xyz() for w in ca_pts])
    c_xyz = np.stack([w.xyz() for w in c_pts])
    n_xyz = np.stack([w.xyz() for w in n_pts]) if n_pts else np.empty((0, 3))

    c_tree = cKDTree(c_xyz)
    n_tree = cKDTree(n_xyz) if len(n_xyz) > 0 else None

    used_c = set()
    used_n = set()

    final_ca: List[WPoint] = []
    final_c:  List[WPoint] = []
    final_n:  List[WPoint] = []

    # 以 CA 的 prob 由大到小，讓較可信的 CA 先分配
    order = np.argsort([-w.prob for w in ca_pts])

    # 小工具：計算總成本（固定 CA、C；N 可為 None）
    def total_cost(ca_vec: np.ndarray, c_vec: np.ndarray, n_vec: Optional[np.ndarray]) -> float:
        
        # C 距離成本
        dc = dist(ca_vec, c_vec)
        cost_c = p.w_dist_c * (dc / max(1e-6, p.r_ca_c_max)) ** p.distance_power

        if n_vec is None:
            return cost_c

        # N 距離成本
        dn = dist(ca_vec, n_vec)
        cost_n = p.w_dist_n * (dn / max(1e-6, p.r_ca_n_max)) ** p.distance_power

        # 角度成本（可關閉）
        if p.use_angle:
            theta = angle_n_ca_c(n_vec, ca_vec, c_vec)
            cost_ang = p.w_angle * angle_penalty(theta, p.angle_target, p.angle_tol)
        else:
            cost_ang = 0.0

        return cost_c + cost_n + cost_ang

    for i in order:
        ca = ca_xyz[i]

        # === Step 1：找 C 候選（半徑內，必要）===
        c_idx_list = c_tree.query_ball_point(ca, r=p.r_ca_c_max) # 使用 KDTree 執行半徑搜尋，加入所有可能點

        # 過濾出距離大於下限的點
        c_idx_list = [j for j in c_idx_list if dist(ca, c_xyz[j]) >= p.r_ca_c_low] 
        
        best_c_idx = None
        best_c_cost = float("inf")

        for j in c_idx_list:
            if p.exclusive_match and j in used_c: # 此 C 點已被其他 CA 使用，則跳過。
                continue
            
            dj = dist(ca, c_xyz[j]) # 計算距離

            # 如果啟用嚴格鍵長，會開啟2次檢查，若不符則跳過這個點    
            if not in_window(dj, p.ca_c_len_lo, p.ca_c_len_hi, p.bondlen_window):
                continue
            
            # 計算 C 距離成本
            cost_c = p.w_dist_c * (dj / max(1e-6, p.r_ca_c_max)) ** p.distance_power
            if cost_c < best_c_cost:
                best_c_cost = cost_c
                best_c_idx = j

        if best_c_idx is None:
            # 找不到合理的 C → 放棄此 CA
            continue

        # === Step 2：在該 C 的前提下找 N（可為 None）===
        chosen_n = None
        best_total = float("inf")

        if n_tree is not None and len(n_xyz) > 0:
            n_idx_list = n_tree.query_ball_point(ca, r=p.r_ca_n_max) # 使用 KDTree 執行半徑搜尋，加入所有可能點

            # 過濾出距離大於下限的點
            n_idx_list = [j for j in n_idx_list if dist(ca, n_xyz[j]) >= p.r_ca_n_low] 
    
            for k in n_idx_list:
                if p.exclusive_match and k in used_n: # 此 C 點已被其他 CA 使用，則跳過。
                    continue
                dk = dist(ca, n_xyz[k]) # 計算距離

                # 如果啟用嚴格鍵長，會開啟2次檢查，若不符則跳過這個點
                if not in_window(dk, p.ca_n_len_lo, p.ca_n_len_hi, p.bondlen_window):
                    continue

                # 計算總成本 (c+n+angle)，若角度未開啟則僅使用c+n
                total = total_cost(ca, c_xyz[best_c_idx], n_xyz[k])
                if total < best_total:
                    best_total = total
                    chosen_n = k

        
        # === 登記結果 ===
        final_ca.append(ca_pts[i])
        final_c.append(c_pts[best_c_idx])

        if chosen_n is not None:
            final_n.append(n_pts[chosen_n])

        # 將已經匹配到的原子登記，避免重複使用
        if p.exclusive_match:
            used_c.add(best_c_idx)
            if chosen_n is not None:
                used_n.add(chosen_n)

    return final_ca, final_n, final_c

# =========================
# 純鍵長匹配：加入可選排序優先權 
# =========================

def greedy_two_step_length_only(
    ca_pts: List[WPoint],
    c_pts: List[WPoint],
    n_pts: List[WPoint],
    p: Params
) -> Tuple[List[WPoint], List[WPoint], List[WPoint]]:
    
    # 若 CA 或 C 點雲為空，則無法進行匹配
    if not ca_pts or not c_pts:
        return [], [], []

    # 將三類點的 xyz 座標轉成 numpy 陣列方便運算
    ca_xyz = np.stack([w.xyz() for w in ca_pts])
    c_xyz = np.stack([w.xyz() for w in c_pts])
    n_xyz = np.stack([w.xyz() for w in n_pts]) if n_pts else np.empty((0, 3))

    # 建立 KDTree（加速鄰近搜尋）
    c_tree = cKDTree(c_xyz)
    n_tree = cKDTree(n_xyz) if len(n_xyz) > 0 else None
    
    # 用來記錄已使用的 C/N 索引（若 exclusive_match=True）
    used_c, used_n = set(), set()
    
    # 儲存最終結果
    final_ca, final_c, final_n = [], [], []

    # CA 依 prob 由大到小排序（高置信度先分配）
    order = np.argsort([-w.prob for w in ca_pts])

    # 最低距離門檻（至少 1Å）
    c_low = max(1.0, getattr(p, "r_ca_c_low", 1.0))
    n_low = max(1.0, getattr(p, "r_ca_n_low", 1.0))

    # 產生排序 key（合格候選之間如何挑）
    def sort_key(prob_value: float, dist_value: float, idx: int):
        if p.select_priority == "distance_then_prob":
            # 距離優先 → (距離小, prob高)
            return (dist_value, -prob_value, idx)
        # 預設：prob優先 → (prob高, 距離小)
        return (-prob_value, dist_value, idx)

    # === 主迴圈：依序為每個 CA 找 C、N ===
    for i in order:
        ca_vec = ca_xyz[i] # 取出當前 CA 的座標向量

        # ---- Step 1: C（必要）----
        
        # 以半徑 r_ca_c_max 搜尋所有可能的 C
        c_idx_list = c_tree.query_ball_point(ca_vec, r=p.r_ca_c_max)
        # 篩掉距離太近 (<1Å) 的點
        c_idx_list = [j for j in c_idx_list if dist(ca_vec, c_xyz[j]) >= c_low]

        qualified_c = [] # 儲存所有符合條件的候選 C
        for j in c_idx_list:
            if p.exclusive_match and j in used_c:
                continue # 已被其他 CA 使用 → 跳過
            dj = dist(ca_vec, c_xyz[j])
            # 若距離在鍵長視窗範圍內，則視為合格
            if p.ca_c_len_lo <= dj <= p.ca_c_len_hi:
                # 儲存候選 (prob, 距離, 索引)
                qualified_c.append((c_pts[j].prob, dj, j))

        if not qualified_c:
            continue  # 視窗內沒有合格 C → 放棄此 CA
        
        # 根據排序 key 選出最佳 C
        qualified_c.sort(key=lambda t: sort_key(t[0], t[1], t[2]))
        chosen_c_idx = qualified_c[0][2]

        # ---- Step 2: N（可省略）----
        chosen_n_idx = None # 若沒有符合的 N → 保持 None
        if n_tree is not None and len(n_xyz) > 0:
            # 以半徑 r_ca_n_max 搜尋 N 候選
            n_idx_list = n_tree.query_ball_point(ca_vec, r=p.r_ca_n_max)
            # 篩掉距離太近 (<1Å) 的點
            n_idx_list = [k for k in n_idx_list if dist(ca_vec, n_xyz[k]) >= n_low]

            qualified_n = []
            for k in n_idx_list:
                if p.exclusive_match and k in used_n:
                    continue # 此 N 已被其他 CA 使用
                dk = dist(ca_vec, n_xyz[k])
                
                # 若距離在鍵長視窗範圍內，則視為合格
                if p.ca_n_len_lo <= dk <= p.ca_n_len_hi:
                    qualified_n.append((n_pts[k].prob, dk, k))
            
            # 若有符合的 N，取排序後最優者
            if qualified_n:
                qualified_n.sort(key=lambda t: sort_key(t[0], t[1], t[2]))
                chosen_n_idx = qualified_n[0][2]

        # ---- 登記結果 ----
        final_ca.append(ca_pts[i])
        final_c.append(c_pts[chosen_c_idx])
        if p.exclusive_match:
            used_c.add(chosen_c_idx)

        if chosen_n_idx is not None:
            final_n.append(n_pts[chosen_n_idx])
            if p.exclusive_match:
                used_n.add(chosen_n_idx)

    return final_ca, final_n, final_c


# =========================
# 對外 API：bond_match（含統計）
# =========================
def bond_match(
    ca_points: Iterable,
    n_points: Iterable,
    c_points: Iterable,
    params: Optional[Params] = None,
    return_triplets: bool = False,
    as_tuple: bool = False,
    return_matching: bool = False,   # ✅ 新增
):
    """
    輸入：三類點（CA/N/C），格式可為 WPoint / 具 x,y,z 的物件 / (x,y,z[,prob])
    輸出：
      - 預設：dict，含 "final_ca/final_n/final_c"（WPoint）與 "stats"
      - return_triplets=True：回傳 (final_ca, final_n, final_c)
      - 再加 as_tuple=True：每類轉為 (x,y,z,prob) 形式，方便寫 CSV
      - ✅ return_matching=True：回傳 mapping {ca_idx: (n_idx, c_idx)}（跟 match_atoms 同格式）
    """
    p_in = params or Params()
    # 防呆 + copy：避免修改外部傳入的 params（也避免 0 做除數）
    p = replace(
        p_in,
        r_ca_c_max=max(1e-6, p_in.r_ca_c_max),
        r_ca_n_max=max(1e-6, p_in.r_ca_n_max),
        angle_tol=max(1e-6, p_in.angle_tol),
    )
    print(f"目前使用的參數為{p_in}")

    # ca = to_wpoints(ca_points)
    # nn = to_wpoints(n_points)
    # cc = to_wpoints(c_points)

    def _dic_xyz_to_list(dic):
        # dic: {idx: "x y z"}
        # 轉成 [(x,y,z), ...] 並保持 sorted key 的順序
        out = []
        for k in sorted(dic.keys()):
            xyz = [float(x) for x in str(dic[k]).split() if x != ""]
            out.append((xyz[0], xyz[1], xyz[2]))
        return out

    # --- 在 bond_match 內，to_wpoints 前面加入 ---
    if isinstance(ca_points, dict):
        ca_points = _dic_xyz_to_list(ca_points)
    if isinstance(n_points, dict):
        n_points = _dic_xyz_to_list(n_points)
    if isinstance(c_points, dict):
        c_points = _dic_xyz_to_list(c_points)

    ca = to_wpoints(ca_points)
    nn = to_wpoints(n_points)
    cc = to_wpoints(c_points)


    # 若未開啟角度，則使用純鍵長匹配
    # if not p.use_angle:
    #     fca, fn, fc = greedy_two_step_length_only(ca, cc, nn, p)
    # else:
    #     fca, fn, fc = greedy_two_step(ca, cc, nn, p)
    
    fca, fn, fc = greedy_two_step(ca, cc, nn, p)

    # ===== ✅ 新增：輸出 mapping（跟 match_atoms 一樣）=====
    if return_matching:
        # 先建立 (x,y,z) -> index 的查表（用 round 避免浮點誤差）
        def key(w, nd=3):
            return (round(float(w.x), nd), round(float(w.y), nd), round(float(w.z), nd))

        ca_index = {key(w): i for i, w in enumerate(ca)}
        n_index  = {key(w): i for i, w in enumerate(nn)}
        c_index  = {key(w): i for i, w in enumerate(cc)}

        # 這裡假設 greedy_two_step 回傳的三個 list 是「對齊的 triplets」
        # 即 fca[k] 對應 fn[k], fc[k]
        matching = {}
        L = min(len(fca), len(fn), len(fc))
        for k in range(L):
            k_ca = key(fca[k])
            k_n  = key(fn[k])
            k_c  = key(fc[k])
            if k_ca in ca_index and k_n in n_index and k_c in c_index:
                matching[ca_index[k_ca]] = (n_index[k_n], c_index[k_c])
        return matching
    
    # 簡單統計
    num_pairs = len(fca)      # 其實就是成功配到 C 的 CA 數
    num_n = len(fn)           # 成功配到 N 的數
    n_ca_ratio = (num_n / num_pairs) if num_pairs > 0 else 0.0

    if return_triplets:
        if as_tuple:
            to_t = lambda L: [(w.x, w.y, w.z, w.prob) for w in L]
            return to_t(fca), to_t(fn), to_t(fc)
        return fca, fn, fc

    return {
        "final_ca": fca,
        "final_n": fn,
        "final_c": fc,
        "stats": {
            "num_pairs": num_pairs,
            "num_n": num_n,
            "n_ca_ratio": n_ca_ratio,
        }
    }


