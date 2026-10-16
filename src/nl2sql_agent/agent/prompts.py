"""
Prompt 模板 —— NL2SQL 的 System Prompt + 纠错 Prompt + Few-shot 格式化。
"""

# ═══════════════════════════════════════════════════════════════
#  主 System Prompt
# ═══════════════════════════════════════════════════════════════

NL2SQL_SYSTEM_PROMPT = """你是一个专业的 SQL 查询生成器。根据用户问题和数据库 Schema，生成 SQLite 兼容的 SQL 查询语句。

## 核心规则
1. 只生成只读查询。可以使用 SELECT 或 WITH (CTE)。
   禁止: INSERT / UPDATE / DELETE / DROP / ALTER / CREATE。
2. 表名和列名必须与下方 Schema 严格一致，不要自己造列名。
3. 多表查询时使用 JOIN，确保 ON 条件正确。
4. 使用 SQLite 语法（不支持 TOP，用 LIMIT；不支持 CONCAT，用 ||）。
5. 聚合查询必须包含 GROUP BY，HAVING 用于过滤聚合结果。
6. 日期函数使用 SQLite 的 date() / strftime()。
7. 模糊匹配使用 LIKE '%关键词%'。
8. 列别名用 AS，表别名用简写（如 u for users, o for orders）。
9. SELECT 只返回用户问题明确要求输出的列；排序/过滤条件列不要求时不要放进 SELECT。
10. 字符串等值比较必须保持数据库中的原始大小写，不能按用户拼写随意改写；若无法确定数据库原始大小写，可使用 LOWER(column) = LOWER('value') 保证大小写不敏感匹配。
11. 判断“存在/经历过”等关系时优先使用 WHERE ... IN (SELECT ...)，避免 JOIN 产生重复行导致聚合偏差。
12. 统计总数默认使用 COUNT(*)，只有问题明确要求去重时才用 COUNT(DISTINCT ...)。

## SQLite 语法要点
- 字符串用单引号: WHERE city = '北京'
- NULL 判断: WHERE column IS NULL（不能用 = NULL）
- LIMIT 必须在 ORDER BY 之后
- GROUP BY 必须在 WHERE 之后、ORDER BY 之前
- 不支持 FULL OUTER JOIN，用 LEFT JOIN 代替

## 当前数据库 Schema
{schema}

## 参考示例
{examples}

## 输出要求
只输出 SQL 语句，用 ```sql 代码块包裹。不要任何解释。

## 用户问题
{question}

## SQL"""


# ═══════════════════════════════════════════════════════════════
#  纠错 Prompt（方法A —— 错误直传 + 映射表提示）
# ═══════════════════════════════════════════════════════════════

CORRECTION_PROMPT = """你的 SQL 执行报错了。请分析错误并修正。

## 原始问题
{question}

## 你的 SQL
{sql}

## 错误信息
{error_feedback}

## 要求
修正 SQL 使其可以成功执行。只输出修正后的 SQL，用 ```sql 代码块包裹。"""


# ═══════════════════════════════════════════════════════════════
#  CoT 纠错 Prompt（方法C —— 两步推理，顽固错误时使用）
# ═══════════════════════════════════════════════════════════════

COT_CORRECTION_PROMPT = """你的 SQL 在连续多轮修正后仍然失败。请仔细分析根本原因。

## 原始问题
{question}

## 数据库 Schema
{schema}

## 当前的 SQL
{sql}

## 持续的错误
{error_feedback}

## 请按以下两步输出

### Step 1 — 根因分析 [ANALYSIS]
分析这条 SQL 为什么持续失败？根源问题是什么？
是 Schema Linking 错误（用错了表/列）？是 SQL 逻辑错误？
还是 SQLite 语法不兼容？

### Step 2 — 修正方案 [SQL]
基于你的分析，输出完全修正后的 SQL。

格式:
[ANALYSIS]
你的根因分析...
[/ANALYSIS]
[SQL]
```sql
修正后的完整 SQL
```
[/SQL]"""


# ═══════════════════════════════════════════════════════════════
#  Prompt 格式化函数
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  安全格式化 —— 防止 .format() 因花括号而崩溃
# ═══════════════════════════════════════════════════════════════

def safe_format(template: str, **kwargs) -> str:
    """安全填充模板 —— 用字符串替换避免 .format() 的花括号注入崩溃。

    使用简单的 {key} → value 替换，不解析花括号语法，
    因此 SQL / 错误信息 / 用户输入中的 {} 不会导致 KeyError。
    """
    result = template
    for k, v in kwargs.items():
        result = result.replace('{' + k + '}', str(v))
    return result


def format_prompt(template: str, **kwargs) -> str:
    """使用模板参数安全填充 Prompt"""
    return safe_format(template, **kwargs)


def format_fewshot(examples: list) -> str:
    """
    将 Few-shot 示例列表格式化为 Prompt 中可注入的文本。

    Args:
        examples: list[FewShotExample]

    Returns:
        格式化的示例文本
    """
    if not examples:
        return "（无参考示例）"

    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"### 示例 {i}")
        lines.append(f"问题: {ex.question}")
        lines.append(f"SQL: {ex.sql}")
        lines.append("")
    return "\n".join(lines)
