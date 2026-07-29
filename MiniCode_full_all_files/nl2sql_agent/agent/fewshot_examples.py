"""
Few-shot 示例池 —— 17 条 (15+2 条 CTE) 覆盖不同难度的示例。

每条示例带标签（difficulty / skills / tables），供 TagBasedFewShotSelector 动态选择。
示例按照预设的电商数据库 Schema 编写:
  users, products, orders, order_items, user_behaviors
"""

from nl2sql_agent.schemas import FewShotExample
from nl2sql_agent.config import Difficulty, SQLSkill

S = SQLSkill  # 简写

# ═══════════════════════════════════════════════════════════════
#  Easy 示例（单表 + 简单条件）
# ═══════════════════════════════════════════════════════════════

EX_EASY_1 = FewShotExample(
    question="价格大于500元的商品有哪些？",
    sql="SELECT name, price FROM products WHERE price > 500 ORDER BY price DESC;",
    difficulty=Difficulty.EASY,
    skills=[S.SINGLE_TABLE, S.ORDER_BY],
    tables=["products"],
    description="单表 + WHERE 条件 + 排序",
)

EX_EASY_2 = FewShotExample(
    question="注册最早的10个用户是谁？",
    sql="SELECT name, register_date FROM users ORDER BY register_date ASC LIMIT 10;",
    difficulty=Difficulty.EASY,
    skills=[S.SINGLE_TABLE, S.ORDER_BY, S.LIMIT],
    tables=["users"],
    description="单表 + 排序 + LIMIT",
)

EX_EASY_3 = FewShotExample(
    question="商品名包含'手机'的商品有哪些？",
    sql="SELECT name, price, category FROM products WHERE name LIKE '%手机%';",
    difficulty=Difficulty.EASY,
    skills=[S.SINGLE_TABLE, S.LIKE],
    tables=["products"],
    description="单表 + LIKE 模糊匹配",
)

EX_EASY_4 = FewShotExample(
    question="北京和上海的用户列表？",
    sql="SELECT name, city FROM users WHERE city IN ('北京', '上海') ORDER BY city;",
    difficulty=Difficulty.EASY,
    skills=[S.SINGLE_TABLE, S.ORDER_BY],
    tables=["users"],
    description="单表 + IN 条件",
)

EX_EASY_5 = FewShotExample(
    question="有哪些商品类别？",
    sql="SELECT DISTINCT category FROM products ORDER BY category;",
    difficulty=Difficulty.EASY,
    skills=[S.SINGLE_TABLE, S.ORDER_BY],
    tables=["products"],
    description="单表 + DISTINCT 去重",
)

# ═══════════════════════════════════════════════════════════════
#  Medium 示例（单表聚合 or 两表 JOIN）
# ═══════════════════════════════════════════════════════════════

EX_MED_1 = FewShotExample(
    question="每个品类的商品数量是多少？",
    sql="SELECT category, COUNT(*) as product_count FROM products GROUP BY category ORDER BY product_count DESC;",
    difficulty=Difficulty.MEDIUM,
    skills=[S.AGGREGATION, S.GROUP_BY, S.ORDER_BY],
    tables=["products"],
    description="单表 + GROUP BY + 聚合计数",
)

EX_MED_2 = FewShotExample(
    question="商品平均价格最高的品类是哪个？",
    sql="SELECT category, AVG(price) as avg_price FROM products GROUP BY category ORDER BY avg_price DESC LIMIT 1;",
    difficulty=Difficulty.MEDIUM,
    skills=[S.AGGREGATION, S.GROUP_BY, S.ORDER_BY, S.LIMIT],
    tables=["products"],
    description="单表 + AVG + GROUP BY + LIMIT",
)

EX_MED_3 = FewShotExample(
    question="每个用户下了多少订单？",
    sql="SELECT u.name, COUNT(o.order_id) as order_count FROM users u JOIN orders o ON u.user_id = o.user_id GROUP BY u.user_id ORDER BY order_count DESC;",
    difficulty=Difficulty.MEDIUM,
    skills=[S.JOIN, S.AGGREGATION, S.GROUP_BY, S.ORDER_BY],
    tables=["users", "orders"],
    description="两表 JOIN + GROUP BY + COUNT",
)

EX_MED_4 = FewShotExample(
    question="2024年1月的订单总额是多少？",
    sql="SELECT SUM(total_amount) as total_revenue FROM orders WHERE order_date >= '2024-01-01' AND order_date < '2024-02-01';",
    difficulty=Difficulty.MEDIUM,
    skills=[S.AGGREGATION, S.DATE_FILTER],
    tables=["orders"],
    description="单表 + SUM + 日期范围过滤",
)

EX_MED_5 = FewShotExample(
    question="哪些用户浏览过商品但从未下单？",
    sql="SELECT DISTINCT u.name FROM users u JOIN user_behaviors ub ON u.user_id = ub.user_id WHERE ub.behavior_type = 'view' AND u.user_id NOT IN (SELECT DISTINCT user_id FROM orders);",
    difficulty=Difficulty.MEDIUM,
    skills=[S.JOIN, S.SUBQUERY],
    tables=["users", "user_behaviors", "orders"],
    description="两表 JOIN + NOT IN 子查询",
)

# ═══════════════════════════════════════════════════════════════
#  Hard 示例（三表 JOIN or 嵌套子查询 or HAVING）
# ═══════════════════════════════════════════════════════════════

EX_HARD_1 = FewShotExample(
    question="北京用户买得最多的商品品类是什么？",
    sql="SELECT p.category, COUNT(*) as cnt FROM order_items oi JOIN orders o ON oi.order_id = o.order_id JOIN users u ON o.user_id = u.user_id JOIN products p ON oi.product_id = p.product_id WHERE u.city = '北京' GROUP BY p.category ORDER BY cnt DESC LIMIT 1;",
    difficulty=Difficulty.HARD,
    skills=[S.JOIN, S.AGGREGATION, S.GROUP_BY, S.ORDER_BY, S.LIMIT],
    tables=["users", "orders", "order_items", "products"],
    description="四表 JOIN + GROUP BY + LIMIT",
)

EX_HARD_2 = FewShotExample(
    question="客单价高于平均客单价的用户有哪些？",
    sql="SELECT u.name, AVG(o.total_amount) as avg_order FROM users u JOIN orders o ON u.user_id = o.user_id GROUP BY u.user_id HAVING AVG(o.total_amount) > (SELECT AVG(total_amount) FROM orders);",
    difficulty=Difficulty.HARD,
    skills=[S.JOIN, S.AGGREGATION, S.GROUP_BY, S.HAVING, S.SUBQUERY],
    tables=["users", "orders"],
    description="两表 JOIN + HAVING + 嵌套子查询（关联子查询）",
)

EX_HARD_3 = FewShotExample(
    question="购买过电子产品的用户还买过什么其他品类？",
    sql="SELECT DISTINCT p2.category FROM order_items oi1 JOIN products p1 ON oi1.product_id = p1.product_id JOIN orders o ON oi1.order_id = o.order_id JOIN order_items oi2 ON o.order_id = oi2.order_id JOIN products p2 ON oi2.product_id = p2.product_id WHERE p1.category = '电子产品' AND p2.category != '电子产品';",
    difficulty=Difficulty.HARD,
    skills=[S.JOIN, S.AGGREGATION],
    tables=["products", "orders", "order_items"],
    description="五表自关联 JOIN + 复杂条件",
)

EX_HARD_4 = FewShotExample(
    question="销售额最高的前5个商品的名称、品类和销售额？",
    sql="SELECT p.name, p.category, SUM(oi.subtotal) as revenue FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.product_id ORDER BY revenue DESC LIMIT 5;",
    difficulty=Difficulty.HARD,
    skills=[S.JOIN, S.AGGREGATION, S.GROUP_BY, S.ORDER_BY, S.LIMIT],
    tables=["products", "order_items"],
    description="两表 JOIN + SUM + GROUP BY + TOP-N",
)

EX_HARD_5 = FewShotExample(
    question="哪些用户把商品加入购物车但从未下单？按加入次数从高到低排列",
    sql="SELECT u.name, COUNT(ub.behavior_id) as cart_count FROM users u JOIN user_behaviors ub ON u.user_id = ub.user_id WHERE ub.behavior_type = 'cart' AND u.user_id NOT IN (SELECT DISTINCT user_id FROM orders) GROUP BY u.user_id ORDER BY cart_count DESC;",
    difficulty=Difficulty.HARD,
    skills=[S.JOIN, S.AGGREGATION, S.GROUP_BY, S.ORDER_BY, S.SUBQUERY],
    tables=["users", "user_behaviors", "orders"],
    description="两表 JOIN + NOT IN 子查询 + GROUP BY + 排序",
)

# ═══════════════════════════════════════════════════════════════
#  Extra: 窗口函数 + CTE（算法工程师高级场景）
# ═══════════════════════════════════════════════════════════════

EX_CTE_1 = FewShotExample(
    question="统计每月GMV总额及环比增长率",
    sql="WITH monthly AS (SELECT strftime('%Y-%m', order_date) AS m, SUM(total_amount) AS gmv FROM orders WHERE status='completed' GROUP BY m) SELECT m, gmv, ROUND((gmv - LAG(gmv) OVER (ORDER BY m)) * 100.0 / LAG(gmv) OVER (ORDER BY m), 2) AS growth_pct FROM monthly ORDER BY m;",
    difficulty=Difficulty.HARD,
    skills=[S.AGGREGATION, S.GROUP_BY, S.DATE_FILTER, S.SUBQUERY],
    tables=["orders"],
    description="CTE + LAG窗口函数: 环比增长率",
)

EX_CTE_2 = FewShotExample(
    question="每个品类销售额Top3商品，用窗口函数排名",
    sql="SELECT category, name, revenue FROM (SELECT p.category, p.name, SUM(oi.subtotal) AS revenue, ROW_NUMBER() OVER (PARTITION BY p.category ORDER BY SUM(oi.subtotal) DESC) AS rn FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.category, p.product_id) WHERE rn <= 3 ORDER BY category, rn;",
    difficulty=Difficulty.HARD,
    skills=[S.JOIN, S.AGGREGATION, S.GROUP_BY, S.ORDER_BY],
    tables=["products", "order_items"],
    description="窗口函数: ROW_NUMBER + PARTITION BY",
)

# ═══════════════════════════════════════════════════════════════
#  完整示例池 (17条: 5 easy + 5 medium + 5 hard + 2 CTE/window)
# ═══════════════════════════════════════════════════════════════

FEWSHOT_POOL: list[FewShotExample] = [
    EX_EASY_1, EX_EASY_2, EX_EASY_3, EX_EASY_4, EX_EASY_5,
    EX_MED_1, EX_MED_2, EX_MED_3, EX_MED_4, EX_MED_5,
    EX_HARD_1, EX_HARD_2, EX_HARD_3, EX_HARD_4, EX_HARD_5,
    EX_CTE_1, EX_CTE_2,
]


def create_default_pool() -> list[FewShotExample]:
    """创建默认的 Few-shot 示例池（17条: 5 easy + 5 medium + 5 hard + 2 CTE/window）"""
    return list(FEWSHOT_POOL)
