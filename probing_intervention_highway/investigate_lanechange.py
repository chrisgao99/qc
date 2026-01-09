import numpy as np

def count_lane_change_windows(lanes: np.ndarray, h: int):
    """
    lanes: 形状 (N,) 的车道索引（例如 0,1,2,...）
    h: 窗口长度（比如 horizon_length = 5）

    返回:
        num_change:  窗口内至少有一次变道的数量
        num_no_change: 窗口内车道始终不变的数量
    """
    lanes = np.asarray(lanes)
    N = lanes.shape[0]
    if N < h:
        return 0, 0

    num_windows = N - h + 1
    num_change = 0
    num_no_change = 0

    # 遍历所有起点 t，窗口是 lanes[t : t+h]
    for t in range(num_windows):
        window = lanes[t:t+h]  # 长度 h
        # 看相邻步之间是否有 lane 发生变化
        if np.any(window[:-1] != window[1:]):
            num_change += 1
        else:
            num_no_change += 1

    return num_change, num_no_change


if __name__ == "__main__":
    # ==== 你可以修改这两个参数 ====
    npz_path = "/p/yufeng/qc/highway_expert_300000_lane.npz"
    h = 7  # 对应 FLAGS.horizon_length
    # ===========================

    data = np.load(npz_path)
    lanes = data["lanes"].astype(np.int32)  # (N,)

    num_change, num_no_change = count_lane_change_windows(lanes, h)
    total = num_change + num_no_change

    print(f"NPZ path: {npz_path}")
    print(f"h (window length) = {h}")
    print(f"Total windows:         {total}")
    print(f"With lane change:      {num_change}  ({num_change / total:.2%} if total>0 else 0)")
    print(f"Without lane change:   {num_no_change}  ({num_no_change / total:.2%} if total>0 else 0)")
