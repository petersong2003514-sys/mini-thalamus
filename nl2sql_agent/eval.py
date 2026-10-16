#!/usr/bin/env python3
"""
NL2SQL Agent — 批量评测

用法:
  python eval.py                          # 跑全部 20 条用例
  python eval.py --difficulty easy        # 只跑 easy 难度
  python eval.py --ids E01,M03,H01        # 跑指定 ID
  python eval.py --output report.json     # 输出 JSON 报告
"""

import sys
import os
import json
import time
import argparse
import re

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
#  评测逻辑
# ═══════════════════════════════════════════════════════════════

def check_execution_success(result) -> bool:
    """SQL 是否成功执行"""
    return result.success and result.result is not None and result.result.success


def check_result_match(result, expected_hint: dict) -> tuple[bool, str]:
    """
    检查执行结果是否符合预期（模糊匹配）。

    支持的 expected_result_hint 类型:
      - exact_row_count: int       → 精确行数匹配
      - min_rows: int              → 最少行数
      - max_rows: int              → 最多行数
      - category: str              → 结果中应包含此值（如 "电子产品"）
      - contains: list[str]        → 结果中应包含所有这些值
      - contains_product: str      → 至少一行包含此产品名（子串匹配）
      - note: str                  → 人工判断标记（不做自动匹配，标注为 SKIP）
    """
    if not check_execution_success(result):
        return False, "SQL 执行失败"

    res = result.result
    hints = expected_hint

    # 扁平化所有结果值
    all_values = []
    for row in res.rows:
        for val in row:
            all_values.append(str(val))

    all_text = " ".join(all_values)

    # note-only 标记 → 跳过自动匹配
    if hints.keys() <= {"note"}:
        return True, "SKIP (manual check: note)"

    # 精确行数
    if "exact_row_count" in hints:
        if res.row_count != hints["exact_row_count"]:
            return False, f"行数不符: 预期 {hints['exact_row_count']}, 实际 {res.row_count}"

    # 最少行数
    if "min_rows" in hints:
        if res.row_count < hints["min_rows"]:
            return False, f"行数过少: 预期 ≥{hints['min_rows']}, 实际 {res.row_count}"

    # 最多行数
    if "max_rows" in hints:
        if res.row_count > hints["max_rows"]:
            return False, f"行数过多: 预期 ≤{hints['max_rows']}, 实际 {res.row_count}"

    # 包含特定值
    if "category" in hints:
        if hints["category"] not in all_values:
            return False, f"结果中未找到预期值: {hints['category']}"

    # 包含列表
    if "contains" in hints:
        for item in hints["contains"]:
            if item not in all_values:
                return False, f"结果中未找到预期值: {item}"

    # 包含产品名（子串匹配）
    if "contains_product" in hints:
        found = any(
            hints["contains_product"].lower() in v.lower()
            for v in all_values
        )
        if not found:
            return False, f"结果中未找到产品: {hints['contains_product']}"

    return True, "OK"


def check_sql_pattern(sql: str, pattern: str) -> tuple[bool, str]:
    """检查 SQL 是否匹配预期正则模式"""
    if not pattern:
        return True, "SKIP (no pattern)"
    try:
        sql_normalized = ' '.join(sql.split())  # 压缩空白
        if re.search(pattern, sql_normalized, re.IGNORECASE):
            return True, "OK"
        return False, f"SQL 不匹配预期模式"
    except re.error as e:
        return False, f"正则错误: {e}"


# ═══════════════════════════════════════════════════════════════
#  报告生成
# ═══════════════════════════════════════════════════════════════

def generate_report(test_cases: list, agent_results: list) -> dict:
    """汇总评测结果为结构化报告"""
    total = len(test_cases)
    exec_ok = sum(1 for r in agent_results if check_execution_success(r))
    match_ok = 0
    pattern_ok = 0

    failures = []
    by_difficulty = {}
    by_stop_reason = {}

    for tc, r in zip(test_cases, agent_results):
        diff = tc["difficulty"]

        # 按难度统计
        if diff not in by_difficulty:
            by_difficulty[diff] = {"count": 0, "exec_ok": 0, "match_ok": 0, "pattern_ok": 0}
        by_difficulty[diff]["count"] += 1

        exec_success = check_execution_success(r)
        if exec_success:
            by_difficulty[diff]["exec_ok"] += 1

        match_success, match_detail = check_result_match(r, tc.get("expected_result_hint", {}))
        if match_success:
            by_difficulty[diff]["match_ok"] += 1
            match_ok += 1

        pattern_success, pattern_detail = check_sql_pattern(r.final_sql, tc.get("expected_sql_pattern", ""))
        if pattern_success:
            by_difficulty[diff]["pattern_ok"] += 1
            pattern_ok += 1

        # 按停止原因统计
        sr = r.stop_reason.value
        by_stop_reason[sr] = by_stop_reason.get(sr, 0) + 1

        # 失败记录
        if not exec_success or not match_success:
            failures.append({
                "id": tc["id"],
                "question": tc["question"],
                "difficulty": diff,
                "exec_success": exec_success,
                "match_success": match_success,
                "match_detail": match_detail,
                "pattern_success": pattern_success,
                "pattern_detail": pattern_detail,
                "stop_reason": r.stop_reason.value,
                "rounds": r.rounds_attempted,
                "sql": r.final_sql[:300] if r.final_sql else "",
                "error": r.result.error if r.result and not r.result.success else "",
            })

    # 计算准确率
    for diff, stats in by_difficulty.items():
        stats["exec_accuracy"] = round(stats["exec_ok"] / stats["count"], 3) if stats["count"] else 0
        stats["match_accuracy"] = round(stats["match_ok"] / stats["count"], 3) if stats["count"] else 0
        stats["pattern_accuracy"] = round(stats["pattern_ok"] / stats["count"], 3) if stats["count"] else 0

    avg_rounds = round(
        sum(r.rounds_attempted for r in agent_results) / total, 2
    ) if total else 0
    avg_time = round(
        sum(r.total_time_ms for r in agent_results) / total, 0
    ) if total else 0
    avg_tokens = round(
        sum(r.total_tokens for r in agent_results) / total, 0
    ) if total else 0

    return {
        "summary": {
            "total": total,
            "execution_success": exec_ok,
            "result_match": match_ok,
            "pattern_match": pattern_ok,
            "execution_accuracy": round(exec_ok / total, 3) if total else 0,
            "result_accuracy": round(match_ok / total, 3) if total else 0,
            "pattern_accuracy": round(pattern_ok / total, 3) if total else 0,
            "avg_rounds": avg_rounds,
            "avg_time_ms": avg_time,
            "avg_tokens": avg_tokens,
        },
        "by_difficulty": by_difficulty,
        "by_stop_reason": by_stop_reason,
        "failures": failures,
    }


# ═══════════════════════════════════════════════════════════════
#  Shell 颜色
# ═══════════════════════════════════════════════════════════════

class C:
    G = '\033[92m'; R = '\033[91m'; Y = '\033[93m'
    C = '\033[96m'; B = '\033[1m'; D = '\033[2m'; E = '\033[0m'

    @staticmethod
    def ok(): return C.s(C.G)
    @staticmethod
    def fail(): return C.s(C.R)
    @staticmethod
    def s(c):
        return c if hasattr(sys.stdout, "isatty") and sys.stdout.isatty() else ""


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    from nl2sql_agent.agent_factory import build_agent, load_dotenv

    parser = argparse.ArgumentParser(description="NL2SQL Agent — 批量评测")
    parser.add_argument("--test-cases", "-t", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "test_cases.json"),
                        help="测试用例 JSON 文件路径")
    parser.add_argument("--difficulty", "-d", type=str, default=None,
                        choices=["easy", "medium", "hard"],
                        help="只跑指定难度")
    parser.add_argument("--ids", type=str, default=None,
                        help="只跑指定 ID，逗号分隔 (如 E01,M03,H01)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出 JSON 报告路径")
    parser.add_argument("--max-rounds", "-r", type=int, default=3,
                        help="最大纠错轮次")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示每条用例的完整 SQL 和返回结果")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="静默模式，只显示最终汇总")
    parser.add_argument("--schema-strategy", type=str, default="full",
                        choices=["full", "keyword", "hybrid"],
                        help="Schema 策略: full=全量 | keyword=关键词 | hybrid=混合精排")
    args = parser.parse_args()

    # 加载测试用例
    print(f"\n加载测试用例: {args.test_cases}")
    with open(args.test_cases, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    # 筛选
    test_cases = all_cases
    if args.difficulty:
        test_cases = [tc for tc in test_cases if tc["difficulty"] == args.difficulty]
    if args.ids:
        ids_set = set(args.ids.split(","))
        test_cases = [tc for tc in test_cases if tc["id"] in ids_set]

    print(f"共 {len(test_cases)} 条用例\n")

    # 初始化 Agent
    load_dotenv()
    print("初始化 Agent...")
    agent, _, _ = build_agent(max_rounds=args.max_rounds, schema_strategy=args.schema_strategy)
    print()

    # 逐条评测
    agent_results = []
    for i, tc in enumerate(test_cases, 1):
        qid = tc["id"]
        question = tc["question"]
        diff = tc["difficulty"]
        tables = ", ".join(tc.get("tables", []))

        print(f"[{i}/{len(test_cases)}] {qid} ({diff}) [{tables}]")
        print(f"  Q: {question}")

        start = time.perf_counter()
        result = agent.run(question)
        elapsed = time.perf_counter() - start
        agent_results.append(result)

        exec_ok = check_execution_success(result)
        match_ok, match_detail = check_result_match(result, tc.get("expected_result_hint", {}))
        pattern_ok, pattern_detail = check_sql_pattern(result.final_sql, tc.get("expected_sql_pattern", ""))

        # 单行结果状态
        exec_icon = f"{C.ok()}EXEC_OK{C.E}" if exec_ok else f"{C.fail()}EXEC_FAIL{C.E}"
        match_icon = f"{C.ok()}MATCH_OK{C.E}" if match_ok else f"{C.fail()}MATCH_FAIL{C.E}"
        pattern_icon = f"{C.ok()}PAT_OK{C.E}" if pattern_ok else f"{C.fail()}PAT_FAIL{C.E}"

        if not args.quiet:
            print(f"  {exec_icon}  {match_icon}  {pattern_icon}  "
                  f"rounds={result.rounds_attempted}  time={elapsed:.1f}s  "
                  f"stop={result.stop_reason.value}")

        # 详细模式：显示 SQL + 返回数据
        if args.verbose and not args.quiet:
            if result.final_sql:
                print(f"  {C.D}── SQL ──{C.E}")
                for line in result.final_sql.strip().split('\n')[:12]:
                    print(f"  {C.D}{line}{C.E}")
            if result.result and result.result.success:
                print(f"  {C.D}── 结果 ({result.result.row_count}行) ──{C.E}")
                if result.result.columns:
                    print(f"  {C.D}{' | '.join(result.result.columns)}{C.E}")
                for row in result.result.rows[:5]:
                    print(f"  {C.D}{' | '.join(str(v) for v in row)}{C.E}")
                if result.result.row_count > 5:
                    print(f"  {C.D}... (+{result.result.row_count - 5}行){C.E}")
            # NL 总结
            if getattr(result, 'nl_summary', ''):
                print(f"  {C.ok()}[总结] {result.nl_summary}{C.E}")

        if not match_ok:
            print(f"  {C.Y}MATCH: {match_detail}{C.E}")
        if not pattern_ok:
            print(f"  {C.Y}PAT: {pattern_detail}{C.E}")

        if result.final_sql and not exec_ok:
            sql_short = result.final_sql[:120].replace('\n', ' ')
            print(f"  {C.D}SQL: {sql_short}...{C.E}")

        if not args.quiet:
            print()

    # 生成报告
    report = generate_report(test_cases, agent_results)

    # 打印汇总
    s = report["summary"]
    print("=" * 60)
    print("  评测汇总")
    print("=" * 60)
    print(f"  总用例:       {s['total']}")
    print(f"  执行成功:     {s['execution_success']}/{s['total']} ({s['execution_accuracy']:.0%})")
    print(f"  结果匹配:     {s['result_match']}/{s['total']} ({s['result_accuracy']:.0%})")
    print(f"  模式匹配:     {s['pattern_match']}/{s['total']} ({s['pattern_accuracy']:.0%})")
    print(f"  平均轮次:     {s['avg_rounds']}")
    print(f"  平均耗时:     {s['avg_time_ms']:.0f}ms")

    print(f"\n  按难度:")
    for diff in ["easy", "medium", "hard"]:
        if diff in report["by_difficulty"]:
            d = report["by_difficulty"][diff]
            print(f"    {diff:8s}: exec={d['exec_accuracy']:.0%}  "
                  f"match={d['match_accuracy']:.0%}  "
                  f"pat={d['pattern_accuracy']:.0%}  "
                  f"({d['count']}条)")

    print(f"\n  按停止原因:")
    for reason, count in sorted(report["by_stop_reason"].items(),
                                 key=lambda x: x[1], reverse=True):
        print(f"    {reason}: {count}")

    if report["failures"]:
        print(f"\n  失败用例 ({len(report['failures'])} 条):")
        for f_item in report["failures"]:
            print(f"    [{f_item['id']}] {f_item['question'][:50]}")
            print(f"      exec={f_item['exec_success']}  match={f_item['match_success']}  "
                  f"stop={f_item['stop_reason']}  rounds={f_item['rounds']}")
            if f_item["match_detail"] and f_item["match_detail"] != "OK":
                print(f"      match: {f_item['match_detail']}")
            if f_item["error"]:
                print(f"      error: {f_item['error'][:120]}")

    # 平均耗时（从 agent_results 计算）
    actual_avg_time = sum(r.total_time_ms for r in agent_results) / len(agent_results) if agent_results else 0
    actual_avg_tokens = sum(r.total_tokens for r in agent_results) / len(agent_results) if agent_results else 0
    print(f"\n  实际平均耗时: {actual_avg_time:.0f}ms/条")
    print(f"  实际平均Token: {actual_avg_tokens:.0f}/条")
    print(f"  估算总Token:   {sum(r.total_tokens for r in agent_results)}")

    # 成功亮点
    if report["failures"]:
        print(f"\n  ── 关键发现 ──")
        # 分析失败集中在哪个难度
        fail_diffs = {}
        for f_item in report["failures"]:
            fail_diffs[f_item["difficulty"]] = fail_diffs.get(f_item["difficulty"], 0) + 1
        for diff, cnt in fail_diffs.items():
            total_diff = report["by_difficulty"].get(diff, {}).get("count", 0)
            print(f"    {diff}: {cnt}/{total_diff} 条失败 — 需优化Few-shot或Prompt")

        # 自纠错效果
        fixed = sum(1 for r in agent_results if r.rounds_attempted > 1 and r.success)
        if fixed > 0:
            print(f"    自纠错成功: {fixed} 条在第2-3轮修正成功")
        else:
            print(f"    自纠错: 全部一次通过(0条需要修正)")

    print("=" * 60)

    # 输出 JSON 报告（默认自动保存）
    output_path = args.output or os.path.join(PROJECT_ROOT, "data", "eval_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {output_path}")

    return report


if __name__ == "__main__":
    main()
