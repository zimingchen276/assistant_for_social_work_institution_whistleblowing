import numpy as np
import pandas as pd
from typing import Tuple

from input import case_input
from score_calculation import summarize
from decision_maker import compute_decision_scores  # 你已有的打分函数（计算 R1..R4）

ACTION_NAMES = [
    "直接举报",
    "匿名举报",
    "继续保持观察，等待时机或者搜集更多证据后再采取行动",
    "保持沉默，等待第三方想办法解决问题",
]

def pick_best_action(scored_df: pd.DataFrame, tol: float = 1e-9) -> Tuple[str, int]:
    """
    从含 R1..R4 的 DataFrame 中返回最佳行动（R 最小）。
    返回: (行动名称, 编号1-4)
    - 若出现并列（在 tol 容差内），返回并列的最小编号。
    - 若 R 全为 NaN，抛出 ValueError。
    """
    r = scored_df.loc[scored_df.index[0], ["R1", "R2", "R3", "R4"]].to_numpy(dtype=float)
    if np.all(np.isnan(r)):
        raise ValueError("无法给出最佳选择：R1..R4 全为 NaN。")

    rmin = np.nanmin(r)
    ties = np.where(np.isclose(r, rmin, atol=tol, rtol=0))[0]  # 0-based
    idx = int(ties[0])  # 并列取编号最小者（如需其它策略这里改）
    return ACTION_NAMES[idx], idx + 1

if __name__ == "__main__":
    # 1) 问卷输入（来自 input.py）
    res = case_input()
    # 2) 指标汇总（U/H/C 等）
    summary = summarize(res)
    # 3) 计算 R1..R4
    scored = compute_decision_scores(summary)
    # 4) 输出最佳选择
    name, no = pick_best_action(scored)
    print("我们认为理性的做法是：", name)
