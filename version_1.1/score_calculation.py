# -*- coding: utf-8 -*-
"""
score_calculation.py
负责指标计算：结局概率、分数阶熵、效用、效用映射U
"""

import numpy as np
import pandas as pd


# ========== 1) 由 s/c 生成结局概率表 ==========
def build_prob_table(s: dict, c: dict) -> pd.DataFrame:
    """
    输入:
      s = {1: s1, 2: s2, 3: s3, 4: s4}  (0~1)
      c = {1: c1, 2: c2, 3: c3, 4: c4}  (0~1)
    输出:
      单行 DataFrame, 含 p1_1..p4_4
    """
    row = {}
    for i in (1, 2, 3, 4):
        si, ci = float(s[i]), float(c[i])
        row[f"p{i}_1"] = si * ci
        row[f"p{i}_2"] = si * (1 - ci)
        row[f"p{i}_3"] = (1 - si) * ci
        row[f"p{i}_4"] = (1 - si) * (1 - ci)
    return pd.DataFrame([row])


# ========== 2) 分数阶熵 ==========
def fractional_entropy(df_probs: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    分数阶熵:
      H_alpha(cols) = sum_j p_j * (-ln p_j)^alpha
    输出:
      DataFrame, 列 H_alpha_1..H_alpha_4
    """
    eps = 1e-12

    def H_for(cols):
        P = df_probs[cols].clip(lower=eps, upper=1.0)
        return (P * (-np.log(P)) ** alpha).sum(axis=1)

    return pd.DataFrame({
        "H_alpha_1": H_for(['p1_1','p1_2','p1_3','p1_4']),
        "H_alpha_2": H_for(['p2_1','p2_2','p2_3','p2_4']),
        "H_alpha_3": H_for(['p3_1','p3_2','p3_3','p3_4']),
        "H_alpha_4": H_for(['p4_1','p4_2','p4_3','p4_4']),
    }, index=df_probs.index)


# ========== 3) 效用计算 C ==========
def compute_utilities(res: dict) -> dict:
    """
    按 Ci = (F_total * P_i) / L_total 计算效用
    输入: res (来自 input.py 的结果字典)
    输出: {C1..C4, P_sum_*}
    """
    F_v, L_c = float(res["F_total"]), float(res["L_total"])

    P_1p = float(res["P_results"]["直接举报"]["sum"])
    P_2p = float(res["P_results"]["匿名举报"]["sum"])
    P_3p = float(res["P_results"]["继续保持观察，等待时机或者搜集更多证据后再采取行动"]["sum"])
    P_4p = float(res["P_results"]["保持沉默，等待第三方想办法解决问题"]["sum"])

    def safe_div(num, den): return num/den if den != 0 else np.nan

    return {
        "C1": safe_div(F_v * P_1p, L_c),
        "C2": safe_div(F_v * P_2p, L_c),
        "C3": safe_div(F_v * P_3p, L_c),
        "C4": safe_div(F_v * P_4p, L_c),
        "P_sum_直": P_1p, "P_sum_匿": P_2p, "P_sum_观": P_3p, "P_sum_默": P_4p,
    }


# ========== 4) C → U 变换 ==========
def transform_utilities(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 alpha 将 C1..C4 映射为 U1..U4：
      alpha == 0.5 -> U = C
      alpha <  0.5 -> U = log(C)   (C<=0 -> NaN)
      alpha >  0.5 -> U = sqrt(C)  (C<0  -> NaN)
    """
    df = summary_df.copy()

    def safe_log(x):  return np.where(x > 0, np.log(x), np.nan)
    def safe_sqrt(x): return np.where(x >= 0, np.sqrt(x), np.nan)

    for i, col in enumerate(['C1', 'C2', 'C3', 'C4'], start=1):
        U_col = f"U{i}"
        df[U_col] = np.where(
            df['alpha'] == 0.5, df[col],
            np.where(
                df['alpha'] < 0.5, safe_log(df[col]),
                safe_sqrt(df[col])
            )
        )
    return df


# ========== 5) 一键汇总 ==========
def summarize(res: dict) -> pd.DataFrame:
    """
    输入: input.py 返回的问卷结果 res
    输出: 单行 DataFrame，含
          p{i}_j, H_alpha_i, alpha, lambda1,
          F_total, L_total, C1..C4, U1..U4
    """
    alpha = float(res["alpha"])
    lambda1 = float(res["lambda1"])

    df_probs = build_prob_table(res["s"], res["c"])
    H_alpha_df = fractional_entropy(df_probs, alpha)
    utilities = compute_utilities(res)

    summary = pd.concat([
        df_probs.reset_index(drop=True),
        H_alpha_df.reset_index(drop=True),
        pd.DataFrame({
            "alpha": [alpha],
            "lambda1": [lambda1],
            "F_total": [res["F_total"]],
            "L_total": [res["L_total"]],
            **utilities
        })
    ], axis=1)

    summary = transform_utilities(summary)
    return summary
