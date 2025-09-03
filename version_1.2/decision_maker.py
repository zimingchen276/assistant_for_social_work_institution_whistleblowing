import re
import numpy as np
import pandas as pd

from input import case_input  # 问卷采集（返回结果字典）
from score_calculation import summarize  # 生成一行汇总指标 DataFrame


def utilization_function_with_probability(summarize_df):
    # 设置最小值eps
    eps = 1e-12

    # 用一个字典保存所有的概率数据
    probability_dict = {
        'p1': summarize_df[['p1_1', 'p1_2', 'p1_3', 'p1_4']],
        'p2': summarize_df[['p2_1', 'p2_2', 'p2_3', 'p2_4']],
        'p3': summarize_df[['p3_1', 'p3_2', 'p3_3', 'p3_4']],
        'p4': summarize_df[['p4_1', 'p4_2', 'p4_3', 'p4_4']]
    }

    # 用一个字典保存所有效用数据
    utility_dict = {
        'U1': summarize_df["U1"],
        'U2': summarize_df["U2"],
        'U3': summarize_df["U3"],
        'U4': summarize_df["U4"]
    }

    # 用一个字典保存每个行为的效用变动
    delta_U = {}
    for i in range(1, 5):
        delta_U[f'U{i}_1'] = -1.37
        delta_U[f'U{i}_2'] = 3.13
        delta_U[f'U{i}_3'] = 1.37
        delta_U[f'U{i}_4'] = -3.13

    # 处理效用和熵的计算
    u_x_a = {}

    # 遍历每个行为
    for i in range(1, 5):
        U = utility_dict[f'U{i}']
        p = probability_dict[f'p{i}']

        # 计算效用变动
        U_variants = [U + delta_U[f'U{i}_{j}'] for j in range(1, 5)]

        # 计算熵 - 注意这里需要使用 alpha 值
        alpha_val = summarize_df['alpha'].iloc[0]  # 获取 alpha 值
        H_variants = [(-np.log(p[f'p{i}_{j}'].clip(lower=eps))) ** alpha_val for j in range(1, 5)]

        # 组合效用与熵
        u_x_a[i] = pd.DataFrame({
            f'u_x_a_{i}_{j}': U_variants[j - 1] * H_variants[j - 1] for j in range(1, 5)
        })

    # 将计算结果合并到原始 DataFrame
    result_df = pd.concat([summarize_df] + [u_x_a[i] for i in range(1, 5)], axis=1)
    return result_df, u_x_a


def compute_decision_scores(summarize_df: pd.DataFrame) -> pd.DataFrame:
    # 调用效用函数
    summarize_df, u_x_a = utilization_function_with_probability(summarize_df)

    # 计算期望效用
    summarize_df['E_U_1'] = sum(u_x_a[1][f"u_x_a_1_{i}"] * summarize_df[f"p1_{i}"] for i in range(1, 5))
    summarize_df['E_U_2'] = sum(u_x_a[2][f"u_x_a_2_{i}"] * summarize_df[f"p2_{i}"] for i in range(1, 5))
    summarize_df['E_U_3'] = sum(u_x_a[3][f"u_x_a_3_{i}"] * summarize_df[f"p3_{i}"] for i in range(1, 5))
    summarize_df['E_U_4'] = sum(u_x_a[4][f"u_x_a_4_{i}"] * summarize_df[f"p4_{i}"] for i in range(1, 5))

    summarize_df['max_EU'] = summarize_df[['E_U_1', 'E_U_2', 'E_U_3', 'E_U_4']].max(axis=1)

    # 确保概率列是数值类型
    summarize_df[['p1_1', 'p1_2', 'p1_3', 'p1_4']] = summarize_df[['p1_1', 'p1_2', 'p1_3', 'p1_4']].apply(pd.to_numeric,
                                                                                                          errors='coerce')
    summarize_df[['p2_1', 'p2_2', 'p2_3', 'p2_4']] = summarize_df[['p2_1', 'p2_2', 'p2_3', 'p2_4']].apply(pd.to_numeric,
                                                                                                          errors='coerce')
    summarize_df[['p3_1', 'p3_2', 'p3_3', 'p3_4']] = summarize_df[['p3_1', 'p3_2', 'p3_3', 'p3_4']].apply(pd.to_numeric,
                                                                                                          errors='coerce')
    summarize_df[['p4_1', 'p4_2', 'p4_3', 'p4_4']] = summarize_df[['p4_1', 'p4_2', 'p4_3', 'p4_4']].apply(pd.to_numeric,
                                                                                                          errors='coerce')

    # 计算方差
    summarize_df['var_x_a_1'] = summarize_df[['p1_1', 'p1_2', 'p1_3', 'p1_4']].var(axis=1)
    summarize_df['var_x_a_2'] = summarize_df[['p2_1', 'p2_2', 'p2_3', 'p2_4']].var(axis=1)
    summarize_df['var_x_a_3'] = summarize_df[['p3_1', 'p3_2', 'p3_3', 'p3_4']].var(axis=1)
    summarize_df['var_x_a_4'] = summarize_df[['p4_1', 'p4_2', 'p4_3', 'p4_4']].var(axis=1)

    # 最大方差
    summarize_df['max_EU_var'] = summarize_df[['var_x_a_1', 'var_x_a_2', 'var_x_a_3', 'var_x_a_4']].max(axis=1)

    print(summarize_df[['var_x_a_1', 'var_x_a_2', 'var_x_a_3', 'var_x_a_4', 'max_EU_var']])

    # 计算 R 分数
    summarize_df["R1"] = (summarize_df['lambda1'] / 2) * (
                summarize_df["H_alpha_1"] + (summarize_df['var_x_a_1'] / summarize_df['max_EU_var'])) - (
                                 1 - summarize_df['lambda1']) * (summarize_df['E_U_1'] / summarize_df['max_EU'])
    summarize_df["R2"] = (summarize_df['lambda1'] / 2) * (
                summarize_df["H_alpha_2"] + (summarize_df['var_x_a_2'] / summarize_df['max_EU_var'])) - (
                                 1 - summarize_df['lambda1']) * (summarize_df['E_U_2'] / summarize_df['max_EU'])
    summarize_df["R3"] = (summarize_df['lambda1'] / 2) * (
                summarize_df["H_alpha_3"] + (summarize_df['var_x_a_3'] / summarize_df['max_EU_var'])) - (
                                 1 - summarize_df['lambda1']) * (summarize_df['E_U_3'] / summarize_df['max_EU'])
    summarize_df["R4"] = (summarize_df['lambda1'] / 2) * (
                summarize_df["H_alpha_4"] + (summarize_df['var_x_a_4'] / summarize_df['max_EU_var'])) - (
                                 1 - summarize_df['lambda1']) * (summarize_df['E_U_4'] / summarize_df['max_EU'])

    # 找到最佳决策
    decision = summarize_df[['R1', 'R2', 'R3', 'R4']].apply(lambda row: row.idxmin(), axis=1)

    # 将决策结果赋值为1、2、3、4
    summarize_df['best_action'] = decision.map({'R1': 1, 'R2': 2, 'R3': 3, 'R4': 4})

    return summarize_df


if __name__ == "__main__":
    # 1) 采集一次问卷（来自 input.py）
    res = case_input()  # 返回字典

    # 2) 生成一行汇总指标（来自 score_calculation.py）
    summary_row = summarize(res)  # DataFrame (1 x 全部指标)

    # 3) 计算 R 分数与推荐
    scored = compute_decision_scores(summary_row)

    # 输出结果
    pd.set_option('display.max_columns', None)
    print("\n== 指标与决策分数（含 R1..R4、best_action） ==")
    print(scored.round(6))