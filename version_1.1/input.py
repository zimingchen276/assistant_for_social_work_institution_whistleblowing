def case_input():
    while True:
        base_case = input("请输入你面临的机构内“吹哨”伦理决策问题：")
        print("你面临的问题是否是：", base_case, "如果是，请输入“Y”，否则请输入“N”")
        base_case_confirm = input().strip().upper()
        if base_case_confirm == "Y":
            results = questionaire_group(base_case)
            return results   # 返回结果字典
        else:
            print("已取消，重新开始。\n")


# —— 通用输入工具 —— #
def get_input_int(prompt: str, lo: int, hi: int) -> int:
    """获取整数输入并校验范围"""
    while True:
        try:
            val = int(input(f"{prompt} ({lo}-{hi}): ").strip())
            if lo <= val <= hi:
                return val
            print(f"输入无效，请输入 {lo} 到 {hi} 之间的整数。")
        except ValueError:
            print("输入无效，请输入整数。")


def ask_block(title: str, prompts: list[str], lo: int, hi: int, transform=None) -> dict:
    """成组提问：返回 {索引: 值} 的字典。可传入 transform 对输入做缩放，如 x/100。"""
    print(title)
    out = {}
    for idx, p in enumerate(prompts, start=1):
        v = get_input_int(p, lo, hi)
        out[idx] = transform(v) if transform else v
    return out


def ask_likert_sum(title: str, items: list[str], lo=1, hi=6) -> tuple[dict, int]:
    """Likert 量表成组提问，同时返回逐题结果与加总分"""
    print(title)
    scores = {}
    for i, it in enumerate(items, start=1):
        scores[i] = get_input_int(it, lo, hi)
    return scores, sum(scores.values())


def ask_rewards_for_action(action_name: str) -> tuple[dict, int]:
    """针对某个行动，询问其收益感知 Likert 题目"""
    P_items = [
        "这会提高我在组织中的声望",
        "这会给我带来职业晋升机会",
        "这会使我获得满意的经济等物质奖励",
        "这会让我觉得我在做正确的事情",
        "这会使我获得心理上的满足或者释怀",
    ]
    return ask_likert_sum(f"【{action_name}】你对以下说法认可程度（1-6）：", P_items)


# —— 业务逻辑 —— #
def questionaire_group(base_case: str = None):
    # 基础参数（0-5）
    alpha_raw = (get_input_int("你能在多大水平上承担选择可能带来的风险？请填写0-5之间的数值。", 0, 5) + 1) / 5
    lambda1_raw = (get_input_int("你在做决策时，对风险的考虑在多大程度上影响你的决定?请填写0-5之间的数值。", 0, 5) + 1) / 5

    # 四种行为标签
    actions = [
        "直接举报",
        "匿名举报",
        "继续保持观察，等待时机或者搜集更多证据后再采取行动",
        "保持沉默，等待第三方想办法解决问题",
    ]

    # 成功率 s（0-100 → 0-1 概率）
    s = ask_block(
        "接下来请持续回答，假设你做出了对应选择，你认为问题被解决的概率有多大，请填写0-100之间的数值:",
        actions,
        0,
        100,
        transform=lambda x: x / 100,
    )

    # 损失率 c（0-100 → 0-1 概率）
    c = ask_block(
        "接下来请持续回答，假设你做出了对应选择，你认为你有多大概率会承受难以承受的损失，请填写0-100之间的数值:",
        actions,
        0,
        100,
        transform=lambda x: x / 100,
    )

    # 公平价值（F）
    F_items = [
        "相关人的行为损害了社会公正",
        "有些人没有得到他们应得的",
        "不公正战胜了公正",
        "遭受了不公正的人没有得到补偿",
        "相关人员做决定时没有力求公正",
    ]
    F_scores, F_v = ask_likert_sum("请回答以下关于“公平/公正”的问题，1=完全不同意，6=完全同意：", F_items)

    # 忠诚价值（L）
    L_items = [
        "举报同事或朋友是一种对关系的背叛",
        "即使举报是正确的，也会削弱我对团队的忠诚感",
        "举报行为会被视为对团队利益的损害",
        "考虑到我和朋友、同事的关系，我会犹豫是否继续举报",
        "即使有不当行为，我仍会优先维护与团队的关系",
        "举报会让我感觉自己不再完全属于这个群体",
    ]
    L_scores, L_c = ask_likert_sum("请继续回答以下关于“忠诚/关系”的问题，1=完全不同意，6=完全同意：", L_items)

    # 四种行为的收益感知 Likert
    P_results = {}
    for act in actions:
        scores, total = ask_rewards_for_action(act)
        P_results[act] = {"scores": scores, "sum": total}

    # —— 汇总结果打包 —— #
    results = {
        "base_case": base_case,
        "alpha": alpha_raw,
        "lambda1": lambda1_raw,
        "s": s,           # dict {1: ,2:,3:,4:}
        "c": c,           # dict
        "F_scores": F_scores,
        "F_total": F_v,
        "L_scores": L_scores,
        "L_total": L_c,
        "P_results": P_results,  # dict {action: {scores:..., sum:...}}
    }

    # 打印展示（可选）
    print("\n—— 输入结果汇总 ——")
    print(results)

    return results

if __name__ == "__main__":
    final_results = case_input()
    print("\n最终返回结果：")
    print(final_results)
