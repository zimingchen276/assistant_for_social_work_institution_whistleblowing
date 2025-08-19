import re
import numpy as np
import pandas as pd

from input import case_input                    # 问卷采集（返回结果字典）
from score_calculation import summarize         # 生成一行汇总指标 DataFrame


def compute_decision_scores(data3: pd.DataFrame) -> pd.DataFrame:
    """
    按 EU-FE 的含义计算（无方差项）：
      R_i = λ * H_i^α  -  (1-λ) * [ E_i / max_a |E_a| ]
    步骤：
      1) 行内读取四个行动的期望效用 U1..U4，并做行内归一化（max |E|）；
      2) 逐行读取 λ（优先 'lambda'，否则 'lambda1'，都没有用 0.5）；
      3) 计算 R1..R4 与 best_action（R 最小）。
    """
    data3 = data3.copy()

    # 列名
    U_cols = ['U1', 'U2', 'U3', 'U4']                          # 每行动的期望效用（每行）
    H_cols = ['H_alpha_1', 'H_alpha_2', 'H_alpha_3', 'H_alpha_4']  # 每行动的分数阶熵（每行）

    # 1) 行内归一化的期望效用
    eps = 1e-12
    E_mat = data3[U_cols].to_numpy(dtype=float)                      # shape: (n, 4)
    max_abs_E_row = np.maximum(np.max(np.abs(E_mat), axis=1), eps)   # 每行四个行动的 |E| 最大值
    E_norm_mat = E_mat / max_abs_E_row[:, None]                      # 行内归一化后的 E

    # 2) 逐行 λ
    if 'lambda' in data3.columns:
        lam = data3['lambda'].to_numpy(dtype=float)                  # shape: (n,)
    elif 'lambda1' in data3.columns:
        lam = data3['lambda1'].to_numpy(dtype=float)
    else:
        lam = np.full((len(data3),), 0.5, dtype=float)

    # 3) 逐行动计算 R_i（EU-FE：无方差项）
    #    R_i = λ * H_i  -  (1-λ) * E_i(norm)
    for i in range(4):
        H_i = data3[H_cols[i]].to_numpy(dtype=float)                 # 每行该行动的 H_U^α
        Ei  = E_norm_mat[:, i]                                       # 该行动归一化 E[u]（行内）
        data3[f'R{i+1}'] = lam * H_i - (1.0 - lam) * Ei

    # 推荐决策（R 最小）
    r_cols = ['R1', 'R2', 'R3', 'R4']
    idxmin_labels = data3[r_cols].idxmin(axis=1)                     # e.g., 'R3'
    data3['best_action'] = idxmin_labels.str.extract(r'R(\d+)').astype(int)

    return data3


if __name__ == "__main__":
    # 1) 采集一次问卷（来自 input.py）
    res = case_input()  # 返回字典  :contentReference[oaicite:2]{index=2}

    # 2) 生成一行汇总指标（来自 score_calculation.py）
    summary_row = summarize(res)  # DataFrame (1 x 全部指标)  :contentReference[oaicite:3]{index=3}

    # 3) 计算 R 分数与推荐
    scored = compute_decision_scores(summary_row)

    # 输出结果
    pd.set_option('display.max_columns', None)
    print("\n== 指标与决策分数（含 R1..R4、best_action） ==")
    print(scored.round(6))
