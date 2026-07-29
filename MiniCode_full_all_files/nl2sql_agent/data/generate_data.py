"""
合成电商数据库 —— 5 张表 + 可控规律注入。

表结构:
  users          500 行   用户信息
  products       200 行   商品信息
  orders        2000 行   订单
  order_items   5000 行   订单明细
  user_behaviors 10000 行  用户行为

可控规律（CITY_CATEGORY_MAP）:
  北京 → 电子产品、上海 → 美妆、广州 → 服装、
  深圳 → 数码配件、杭州 → 食品饮料
  → 每个城市 70% 的订单集中在对应品类上
  → 用于自动化评测验证：问"北京用户买最多的品类"→ 答案必为"电子产品"

运行方式:
  cd d:/实习/nl2sql_agent
  python data/generate_data.py
  → 生成 data/ecommerce.db
"""

import sys
import io

# Windows 控制台 UTF-8 编码修复
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sqlite3
import random
import os
from datetime import datetime, timedelta

# 固定随机种子 → 可复现
random.seed(42)

# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════

DB_PATH = os.path.join(os.path.dirname(__file__), "ecommerce.db")

# 城市 → 对应品类（可控规律，评测基准）
CITY_CATEGORY_MAP = {
    "北京": "电子产品",
    "上海": "美妆",
    "广州": "服装",
    "深圳": "数码配件",
    "杭州": "食品饮料",
}
CITIES = list(CITY_CATEGORY_MAP.keys())
CATEGORIES = list(CITY_CATEGORY_MAP.values())

USER_TIERS = ["普通", "白银", "黄金", "铂金"]
PAYMENT_METHODS = ["微信支付", "支付宝", "银行卡", "Apple Pay"]
BEHAVIOR_TYPES = ["view", "cart", "fav", "search"]
ORDER_STATUSES = ["completed", "cancelled", "refunded", "pending"]

# 行数
N_USERS = 500
N_PRODUCTS = 200
N_ORDERS = 2000
N_ORDER_ITEMS = 5000
N_BEHAVIORS = 10000

# 日期范围
DATE_START = datetime(2024, 1, 1)
DATE_END = datetime(2024, 6, 30)

# 品类偏好概率: 用户所在城市的对应品类权重提升
HOME_CATEGORY_WEIGHT = 0.70  # 70% 概率买本城市对应品类


# ═══════════════════════════════════════════════════════════════
#  商品数据 (硬编码真实商品名)
# ═══════════════════════════════════════════════════════════════

PRODUCT_POOL = {
    "电子产品": [
        ("iPhone 15 Pro Max", "Apple", 8999),
        ("iPhone 15", "Apple", 5999),
        ("MacBook Pro 14", "Apple", 14999),
        ("MacBook Air 15", "Apple", 10499),
        ("iPad Air", "Apple", 4399),
        ("iPad Pro", "Apple", 6799),
        ("AirPods Pro", "Apple", 1899),
        ("Apple Watch Ultra", "Apple", 6299),
        ("华为 Mate 60 Pro", "华为", 6999),
        ("华为 P70 Pro", "华为", 5488),
        ("华为 MatePad", "华为", 3299),
        ("荣耀 Magic6", "荣耀", 4399),
        ("小米 14 Ultra", "小米", 5999),
        ("小米平板 6 Pro", "小米", 2499),
        ("OPPO Find X7", "OPPO", 4599),
        ("vivo X100 Pro", "vivo", 4999),
        ("三星 Galaxy S24", "三星", 7499),
        ("索尼 WH-1000XM5", "索尼", 2299),
        ("Bose QC45", "Bose", 1999),
        ("Kindle Paperwhite", "Amazon", 1068),
        ("DJI Mini 4 Pro", "DJI", 4788),
        ("GoPro Hero 12", "GoPro", 3198),
        ("任天堂 Switch OLED", "任天堂", 2599),
        ("PS5 Slim", "索尼", 3299),
        ("Xbox Series X", "微软", 3899),
        ("戴尔 XPS 15", "Dell", 12999),
        ("ThinkPad X1 Carbon", "联想", 9999),
        ("华为 MateBook X Pro", "华为", 10999),
        ("罗技 MX Master 3S", "罗技", 699),
        ("机械革命蛟龙16", "机械革命", 6999),
        ("佳能 EOS R6", "佳能", 15599),
        ("尼康 Z6 II", "尼康", 12999),
        ("大疆 Pocket 3", "DJI", 3499),
        ("Insta360 X4", "Insta360", 3299),
        ("Apple TV 4K", "Apple", 1099),
        ("小米电视 S85", "小米", 7999),
        ("华为智慧屏 V75", "华为", 9999),
        ("Herman Miller Aeron", "Herman Miller", 8999),
        ("SteelSeries Arctis", "SteelSeries", 1199),
        ("雷蛇 蝰蛇 V3", "雷蛇", 699),
    ],
    "美妆": [
        ("兰蔻小黑瓶精华", "兰蔻", 1080),
        ("雅诗兰黛小棕瓶", "雅诗兰黛", 990),
        ("SK-II 神仙水", "SK-II", 1590),
        ("海蓝之谜面霜", "海蓝之谜", 2850),
        ("赫莲娜绿宝瓶", "赫莲娜", 1680),
        ("资生堂红腰子", "资生堂", 860),
        ("科颜氏高保湿霜", "科颜氏", 330),
        ("倩碧黄油", "倩碧", 295),
        ("MAC 子弹头口红", "MAC", 210),
        ("YSL 小金条", "YSL", 370),
        ("TF 黑管唇膏", "Tom Ford", 450),
        ("迪奥 999", "Dior", 390),
        ("阿玛尼红管唇釉", "阿玛尼", 340),
        ("纪梵希小羊皮", "纪梵希", 350),
        ("娇兰复原蜜", "娇兰", 1180),
        ("CPB 隔离霜", "CPB", 540),
        ("NARS 腮红", "NARS", 280),
        ("植村秀卸妆油", "植村秀", 390),
        ("贝德玛卸妆水", "贝德玛", 158),
        ("美宝莲眼唇卸", "美宝莲", 59),
        ("3CE 眼影盘", "3CE", 239),
        ("完美日记唇釉", "完美日记", 69),
        ("花西子散粉", "花西子", 169),
        ("薇诺娜特护霜", "薇诺娜", 268),
        ("至本洗面奶", "至本", 55),
        ("旁氏米粹", "旁氏", 29),
        ("欧莱雅紫熨斗", "欧莱雅", 319),
        ("OLAY 小白瓶", "OLAY", 339),
        ("露得清A醇", "露得清", 239),
        ("The Ordinary 烟酰胺", "The Ordinary", 65),
        ("修丽可色修精华", "修丽可", 680),
        ("理肤泉 B5", "理肤泉", 119),
        ("雅漾喷雾", "雅漾", 98),
        ("珂润面霜", "珂润", 178),
        ("Freeplus 洁面", "Freeplus", 129),
        ("EltaMD 洁面", "EltaMD", 178),
        ("Farmacy 卸妆膏", "Farmacy", 258),
        ("CT 眼影盘", "Charlotte Tilbury", 420),
        ("衰败城市 Naked", "Urban Decay", 390),
        ("芭比布朗虫草粉底", "Bobbi Brown", 520),
    ],
    "服装": [
        ("优衣库羊绒衫", "优衣库", 299),
        ("优衣库轻薄羽绒", "优衣库", 499),
        ("ZARA 西装外套", "ZARA", 699),
        ("ZARA 阔腿裤", "ZARA", 329),
        ("H&M 基础T恤", "H&M", 79),
        ("Nike Air Force 1", "Nike", 849),
        ("Nike Dunk Low", "Nike", 799),
        ("Adidas Samba", "Adidas", 899),
        ("Adidas Ultraboost", "Adidas", 1199),
        ("New Balance 530", "New Balance", 719),
        ("Converse Chuck 70", "Converse", 569),
        ("Vans Old Skool", "Vans", 499),
        ("李宁 超轻21", "李宁", 599),
        ("安踏 C37", "安踏", 399),
        ("FILA 老爹鞋", "FILA", 799),
        ("始祖鸟 Beta AR", "始祖鸟", 5890),
        ("北面 1996 羽绒服", "北面", 2698),
        ("Canada Goose Expedition", "Canada Goose", 9900),
        ("Moncler Maya", "Moncler", 12800),
        ("优衣库修身牛仔裤", "优衣库", 299),
        ("Levi's 501", "Levi's", 799),
        ("太平鸟连衣裙", "太平鸟", 459),
        ("ONLY 风衣", "ONLY", 699),
        ("VERO MODA 衬衫", "VERO MODA", 399),
        ("雅戈尔西服", "雅戈尔", 2299),
        ("波司登羽绒服", "波司登", 1599),
        ("海澜之家 Polo", "海澜之家", 168),
        ("Puma 卫衣", "Puma", 329),
        ("Under Armour 紧身衣", "Under Armour", 349),
        ("lululemon Align", "lululemon", 850),
        ("lululemon Define", "lululemon", 1080),
        ("Mackage 皮衣", "Mackage", 7900),
        ("Patagonia 抓绒", "Patagonia", 1299),
        ("COS 廓形大衣", "COS", 1690),
        ("Massimo Dutti 羊绒围巾", "Massimo Dutti", 590),
        ("蕉下防晒衣", "蕉下", 229),
        ("Ubras 内衣", "Ubras", 129),
        ("NEIWAI 家居服", "NEIWAI", 299),
        ("歌莉娅羽绒马甲", "歌莉娅", 899),
        ("伊芙丽西装裤", "伊芙丽", 499),
    ],
    "数码配件": [
        ("Anker 20W 充电头", "Anker", 49),
        ("Anker 充电宝", "Anker", 149),
        ("Anker 数据线", "Anker", 29),
        ("小米充电宝", "小米", 99),
        ("小米无线充电器", "小米", 79),
        ("倍思车载充电器", "倍思", 39),
        ("绿联 USB Hub", "绿联", 89),
        ("绿联 HDMI 线", "绿联", 35),
        ("罗技 K380 键盘", "罗技", 239),
        ("罗技 Pebble 鼠标", "罗技", 109),
        ("妙控键盘", "Apple", 1399),
        ("Magic Mouse", "Apple", 699),
        ("AirTag", "Apple", 229),
        ("三星 T7 移动硬盘", "三星", 599),
        ("闪迪 U盘 128G", "闪迪", 79),
        ("金士顿 SD卡", "金士顿", 129),
        ("西部数据移动硬盘", "WD", 459),
        ("AirPods 保护壳", "第三方", 29),
        ("手机防窥膜", "第三方", 39),
        ("平板支架", "第三方", 59),
        ("笔记本散热架", "第三方", 89),
        ("屏幕清洁套装", "第三方", 25),
        ("网线 Cat6", "第三方", 19),
        ("Type-C 拓展坞", "绿联", 259),
        ("蓝牙音箱 JBL Go", "JBL", 229),
        ("Marshall Emberton", "Marshall", 1199),
        ("HomePod mini", "Apple", 749),
        ("小爱音箱", "小米", 199),
        ("天猫精灵", "阿里", 99),
        ("AirPods Max 耳罩", "第三方", 39),
        ("Apple Pencil 笔尖", "第三方", 25),
        ("Switch 收纳包", "第三方", 69),
        ("PS5 手柄充电座", "第三方", 99),
        ("相机清洁套装", "第三方", 49),
        ("手机防水袋", "第三方", 19),
        ("GoPro 自拍杆", "第三方", 89),
        ("DJI 收纳包", "DJI", 199),
        ("无线耳机套", "第三方", 15),
        ("智能插座", "小米", 59),
        ("智能灯泡", "小米", 79),
    ],
    "食品饮料": [
        ("三只松鼠坚果礼盒", "三只松鼠", 99),
        ("良品铺子零食大礼包", "良品铺子", 128),
        ("百草味肉脯", "百草味", 39),
        ("来伊份话梅", "来伊份", 25),
        ("旺旺大礼包", "旺旺", 69),
        ("奥利奥饼干", "奥利奥", 15),
        ("费列罗巧克力", "费列罗", 89),
        ("德芙巧克力礼盒", "德芙", 79),
        ("星巴克咖啡豆", "星巴克", 138),
        ("雀巢速溶咖啡", "雀巢", 49),
        ("三顿半咖啡", "三顿半", 189),
        ("瑞幸挂耳咖啡", "瑞幸", 69),
        ("农夫山泉矿泉水", "农夫山泉", 2),
        ("元气森林苏打水", "元气森林", 5),
        ("东方树叶茉莉花茶", "农夫山泉", 5),
        ("康师傅冰红茶", "康师傅", 3),
        ("蒙牛纯牛奶", "蒙牛", 69),
        ("伊利安慕希", "伊利", 59),
        ("简爱酸奶", "简爱", 12),
        ("妙可蓝多奶酪棒", "妙可蓝多", 49),
        ("好想你红枣", "好想你", 59),
        ("蜜雪冰城代金券", "蜜雪冰城", 10),
        ("喜茶代金券", "喜茶", 30),
        ("奈雪的茶礼盒", "奈雪", 168),
        ("茅台迎宾酒", "茅台", 299),
        ("五粮液 52度", "五粮液", 1099),
        ("青岛啤酒 24罐", "青岛啤酒", 89),
        ("科罗娜啤酒", "科罗娜", 15),
        ("拉菲传奇", "拉菲", 159),
        ("黄飞鸿花生", "黄飞鸿", 15),
        ("乐事薯片", "乐事", 8),
        ("卫龙辣条", "卫龙", 5),
        ("海底捞火锅底料", "海底捞", 25),
        ("老干妈辣酱", "老干妈", 12),
        ("五常大米 5kg", "北大荒", 79),
        ("金龙鱼调和油", "金龙鱼", 89),
        ("海天酱油", "海天", 15),
        ("李锦记蚝油", "李锦记", 12),
        ("郫县豆瓣酱", "鹃城", 9),
        ("太太乐鸡精", "太太乐", 8),
    ],
}

# 品牌列表
ALL_BRANDS = set()
for items in PRODUCT_POOL.values():
    for _, brand, _ in items:
        ALL_BRANDS.add(brand)

# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

def random_date(start=DATE_START, end=DATE_END):
    """生成随机日期时间"""
    delta = end - start
    random_days = random.randint(0, delta.days)
    random_seconds = random.randint(0, 86400 - 1)
    return start + timedelta(days=random_days, seconds=random_seconds)


def random_phone():
    """生成随机手机号"""
    prefixes = ["138", "139", "150", "151", "152", "158", "159",
                "186", "187", "188", "130", "131", "132", "155", "156"]
    return random.choice(prefixes) + ''.join(random.choices("0123456789", k=8))


def random_name():
    """生成随机中文名"""
    surnames = ["王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴",
                "徐", "孙", "马", "胡", "朱", "郭", "何", "林", "罗", "高",
                "梁", "郑", "谢", "宋", "唐", "韩", "冯", "于", "董", "萧"]
    given_names = ["明", "华", "强", "伟", "芳", "敏", "静", "丽", "洋", "勇",
                   "军", "杰", "磊", "涛", "鑫", "文", "超", "波", "宁", "鹏",
                   "雪", "琳", "瑶", "怡", "婷", "欣", "雨", "晨", "思", "悦",
                   "建国", "志强", "秀英", "桂英", "文博", "浩然", "子涵", "梓轩"]
    return random.choice(surnames) + random.choice(given_names)


def compute_cost(price):
    """根据售价估算成本（60-80% 的售价）"""
    return round(price * random.uniform(0.6, 0.8), 2)


# ═══════════════════════════════════════════════════════════════
#  数据生成
# ═══════════════════════════════════════════════════════════════

def generate_products():
    """生成 200 个商品"""
    products = []
    pid = 1

    # 每个品类 40 个商品
    for category, items in PRODUCT_POOL.items():
        # 先用预定义的商品名
        for name, brand, price in items:
            products.append({
                "product_id": pid,
                "name": name,
                "category": category,
                "brand": brand,
                "price": price,
                "cost": compute_cost(price),
                "stock": random.randint(10, 500),
                "status": "active",
            })
            pid += 1

    return products


def generate_users():
    """生成 500 个用户，均匀分布在 5 个城市"""
    users = []
    for uid in range(1, N_USERS + 1):
        city = CITIES[(uid - 1) % len(CITIES)]  # 均匀分布

        # 注册日期：2023-2024 均匀分布
        register_date = random_date(
            datetime(2023, 1, 1), datetime(2024, 5, 31)
        )

        # 最后登录：注册日期到 2024-06-30 之间
        last_login = random_date(register_date, DATE_END)

        users.append({
            "user_id": uid,
            "name": random_name(),
            "phone": random_phone(),
            "city": city,
            "tier": random.choices(
                USER_TIERS, weights=[0.5, 0.3, 0.15, 0.05]
            )[0],
            "register_date": register_date.strftime("%Y-%m-%d"),
            "last_login": last_login.strftime("%Y-%m-%d %H:%M:%S"),
        })
    return users


def generate_orders(users, products):
    """
    生成 2000 个订单。

    可控规律注入:
      每个城市的用户有 70% 概率购买本城市对应品类的商品。
      北京 → 电子产品、上海 → 美妆、广州 → 服装、
      深圳 → 数码配件、杭州 → 食品饮料

    每个用户产生 2-8 个订单。
    """
    # 按品类索引商品
    products_by_category = {}
    for p in products:
        products_by_category.setdefault(p["category"], []).append(p)

    # 用户 ID → 用户信息
    user_map = {u["user_id"]: u for u in users}

    orders = []
    order_items = []
    oid = 1
    iid = 1

    # 用户 491-500 是"只看不买"用户，不生成订单
    GHOST_USERS = set(range(491, 501))

    for uid in range(1, N_USERS + 1):
        if uid in GHOST_USERS:
            continue  # 跳过"只看不买"用户

        user = user_map[uid]
        city = user["city"]
        home_category = CITY_CATEGORY_MAP[city]

        n_orders = random.randint(2, 8)

        for _ in range(n_orders):
            order_date = random_date(
                max(DATE_START, datetime.strptime(user["register_date"], "%Y-%m-%d")),
                DATE_END,
            )

            # 生成订单明细（1-5 个商品）
            n_items = random.randint(1, 5)
            item_list = []

            for _ in range(n_items):
                # 70% 概率选本城市的对应品类
                if random.random() < HOME_CATEGORY_WEIGHT:
                    pool = products_by_category[home_category]
                else:
                    pool = products

                product = random.choice(pool)
                quantity = random.randint(1, 3)
                unit_price = product["price"]
                subtotal = quantity * unit_price

                item_list.append({
                    "item_id": iid,
                    "product_id": product["product_id"],
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "subtotal": subtotal,
                })
                iid += 1

            total_amount = sum(item["subtotal"] for item in item_list)
            discount = round(random.uniform(0, total_amount * 0.2), 2)
            final_amount = round(total_amount - discount, 2)

            orders.append({
                "order_id": oid,
                "user_id": uid,
                "order_date": order_date.strftime("%Y-%m-%d %H:%M:%S"),
                "total_amount": final_amount,
                "discount_amount": discount,
                "payment_method": random.choice(PAYMENT_METHODS),
                "status": random.choices(
                    ORDER_STATUSES,
                    weights=[0.75, 0.1, 0.08, 0.07]
                )[0],
            })

            # 关联 order_items
            for item in item_list:
                item["order_id"] = oid
                order_items.append(item)

            oid += 1

    return orders, order_items


def generate_behaviors(users, products, orders):
    """
    生成 10000 条用户行为。

    特殊注入：
      - 10 个"只看不买"用户（user_id 491-500），只有行为、没有订单
        → 用于测试 NOT IN 子查询: "哪些用户浏览过但从未下单？"
    """
    # 找出有订单的用户
    users_with_orders = set(o["user_id"] for o in orders)

    behaviors = []
    bid = 1

    all_product_ids = [p["product_id"] for p in products]

    # 优先为"只看不买"用户 (491-500) 生成行为，确保他们的行为被收录
    GHOST_USERS = list(range(491, 501))
    REGULAR_USERS = [u for u in range(1, N_USERS + 1) if u not in GHOST_USERS]
    user_order = GHOST_USERS + REGULAR_USERS  # 只看不买用户优先

    for uid in user_order:
        n_behaviors = random.randint(20, 40) if uid in GHOST_USERS else random.randint(10, 30)

        for _ in range(n_behaviors):
            behavior_type = random.choices(
                BEHAVIOR_TYPES,
                weights=[0.55, 0.15, 0.10, 0.20]  # view最多, cart/fav少
            )[0]

            product_id = random.choice(all_product_ids)
            behavior_time = random_date()

            # 只看不买用户增加加购行为概率
            if uid in GHOST_USERS:
                behavior_type = random.choices(
                    BEHAVIOR_TYPES,
                    weights=[0.35, 0.35, 0.10, 0.20]  # cart 比例提升
                )[0]

            behaviors.append({
                "behavior_id": bid,
                "user_id": uid,
                "product_id": product_id,
                "behavior_type": behavior_type,
                "behavior_time": behavior_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_sec": (
                    random.randint(5, 300) if behavior_type == "view"
                    else random.randint(1, 60)
                ),
            })

            bid += 1
            if bid > N_BEHAVIORS:
                break

        if bid > N_BEHAVIORS:
            break

    return behaviors[:N_BEHAVIORS]


def add_table_comments(cursor):
    """为表和列添加中文注释（SQLite 不支持 COMMENT，写入 sqlite_comment 表或直接用 PRAGMA）"""
    # SQLite 不原生支持列注释，我们创建一个元数据表来存储
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS _table_descriptions (
            table_name TEXT PRIMARY KEY,
            description TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS _column_descriptions (
            table_name TEXT,
            column_name TEXT,
            description TEXT,
            PRIMARY KEY (table_name, column_name)
        )
    """)

    table_descs = {
        "users": "用户信息表，存储注册用户的基本信息和会员等级",
        "products": "商品信息表，存储所有在售商品的名称、品类、品牌、价格和库存",
        "orders": "订单表，记录用户的每笔订单，包含金额、折扣、支付方式和状态",
        "order_items": "订单明细表，记录每笔订单中购买的具体商品、数量和金额",
        "user_behaviors": "用户行为表，记录用户在平台上的浏览、加购、收藏和搜索行为",
    }

    column_descs = {
        "users": [
            ("user_id", "用户唯一ID"),
            ("name", "用户姓名"),
            ("phone", "手机号"),
            ("city", "所在城市"),
            ("tier", "会员等级（普通/白银/黄金/铂金）"),
            ("register_date", "注册日期"),
            ("last_login", "最后登录时间"),
        ],
        "products": [
            ("product_id", "商品唯一ID"),
            ("name", "商品名称"),
            ("category", "商品品类"),
            ("brand", "品牌"),
            ("price", "售价（元）"),
            ("cost", "成本价（元）"),
            ("stock", "库存数量"),
            ("status", "商品状态（active/inactive）"),
        ],
        "orders": [
            ("order_id", "订单唯一ID"),
            ("user_id", "下单用户ID，关联 users 表"),
            ("order_date", "下单时间"),
            ("total_amount", "实付金额（元）"),
            ("discount_amount", "折扣金额（元）"),
            ("payment_method", "支付方式"),
            ("status", "订单状态（completed/cancelled/refunded/pending）"),
        ],
        "order_items": [
            ("item_id", "明细唯一ID"),
            ("order_id", "所属订单ID，关联 orders 表"),
            ("product_id", "商品ID，关联 products 表"),
            ("quantity", "购买数量"),
            ("unit_price", "商品单价（元）"),
            ("subtotal", "小计金额（元）= quantity × unit_price"),
        ],
        "user_behaviors": [
            ("behavior_id", "行为唯一ID"),
            ("user_id", "用户ID，关联 users 表"),
            ("product_id", "商品ID，关联 products 表"),
            ("behavior_type", "行为类型（view/cart/fav/search）"),
            ("behavior_time", "行为发生时间"),
            ("duration_sec", "停留时长（秒）"),
        ],
    }

    for table, desc in table_descs.items():
        cursor.execute(
            "INSERT OR REPLACE INTO _table_descriptions VALUES (?, ?)",
            (table, desc),
        )
    for table, cols in column_descs.items():
        for col_name, col_desc in cols:
            cursor.execute(
                "INSERT OR REPLACE INTO _column_descriptions VALUES (?, ?, ?)",
                (table, col_name, col_desc),
            )


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  电商数据库合成脚本")
    print("=" * 60)

    # ── Step 1: 生成数据 ──
    print("\n[1/5] 生成商品数据 (200 件)...")
    products = generate_products()
    print(f"  → 生成 {len(products)} 件商品, {len(CATEGORIES)} 个品类")

    print("[2/5] 生成用户数据 (500 人)...")
    users = generate_users()
    city_counts = {}
    for u in users:
        city_counts[u["city"]] = city_counts.get(u["city"], 0) + 1
    print(f"  → 生成 {len(users)} 个用户, 分布: {city_counts}")

    print("[3/5] 生成订单 + 订单明细...")
    orders, order_items = generate_orders(users, products)
    print(f"  → {len(orders)} 个订单, {len(order_items)} 条明细")
    print(f"  → 可控规律已注入 (CITY_CATEGORY_MAP)")

    print("[4/5] 生成用户行为数据...")
    behaviors = generate_behaviors(users, products, orders)
    print(f"  → {len(behaviors)} 条行为记录")
    no_order_users = set(range(491, 501))
    print(f"  → 注入了 {len(no_order_users)} 个'只看不买'用户 (ID 491-500)")

    # ── Step 2: 写入 SQLite ──
    print(f"\n[5/5] 写入 SQLite → {DB_PATH}")

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 建表
    cursor.execute("""
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            phone TEXT,
            city TEXT NOT NULL,
            tier TEXT DEFAULT '普通',
            register_date TEXT NOT NULL,
            last_login TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            brand TEXT,
            price REAL NOT NULL,
            cost REAL,
            stock INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        )
    """)
    cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            total_amount REAL NOT NULL,
            discount_amount REAL DEFAULT 0,
            payment_method TEXT,
            status TEXT DEFAULT 'completed',
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            subtotal REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE user_behaviors (
            behavior_id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            behavior_type TEXT NOT NULL,
            behavior_time TEXT NOT NULL,
            duration_sec INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
    """)

    # 写入数据
    cursor.executemany(
        "INSERT INTO users VALUES (:user_id, :name, :phone, :city, :tier, :register_date, :last_login)",
        users,
    )
    cursor.executemany(
        "INSERT INTO products VALUES (:product_id, :name, :category, :brand, :price, :cost, :stock, :status)",
        products,
    )
    cursor.executemany(
        "INSERT INTO orders VALUES (:order_id, :user_id, :order_date, :total_amount, :discount_amount, :payment_method, :status)",
        orders,
    )
    cursor.executemany(
        "INSERT INTO order_items VALUES (:item_id, :order_id, :product_id, :quantity, :unit_price, :subtotal)",
        order_items,
    )
    cursor.executemany(
        "INSERT INTO user_behaviors VALUES (:behavior_id, :user_id, :product_id, :behavior_type, :behavior_time, :duration_sec)",
        behaviors,
    )

    # 添加注释
    add_table_comments(cursor)

    # 创建索引
    cursor.execute("CREATE INDEX idx_orders_user ON orders(user_id)")
    cursor.execute("CREATE INDEX idx_orders_date ON orders(order_date)")
    cursor.execute("CREATE INDEX idx_order_items_order ON order_items(order_id)")
    cursor.execute("CREATE INDEX idx_order_items_product ON order_items(product_id)")
    cursor.execute("CREATE INDEX idx_behaviors_user ON user_behaviors(user_id)")
    cursor.execute("CREATE INDEX idx_behaviors_product ON user_behaviors(product_id)")
    cursor.execute("CREATE INDEX idx_behaviors_type ON user_behaviors(behavior_type)")
    cursor.execute("CREATE INDEX idx_users_city ON users(city)")
    cursor.execute("CREATE INDEX idx_products_category ON products(category)")

    conn.commit()

    # ── Step 3: 验证 ──
    print("\n" + "=" * 60)
    print("  数据验证")
    print("=" * 60)

    for table in ["users", "products", "orders", "order_items", "user_behaviors"]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table:20s}: {count:>6d} 行")

    # 验证可控规律
    cursor.execute("""
        SELECT u.city, p.category, COUNT(*) as cnt
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN users u ON o.user_id = u.user_id
        JOIN products p ON oi.product_id = p.product_id
        GROUP BY u.city, p.category
        ORDER BY u.city, cnt DESC
    """)
    print("\n  验证 CITY_CATEGORY_MAP 规律 (每个城市购买最多的品类):")
    current_city = None
    for row in cursor.fetchall():
        city, category, cnt = row
        if city != current_city:
            expected = CITY_CATEGORY_MAP.get(city, "?")
            match = "[MATCH]" if category == expected else "[MISMATCH]"
            print(f"    {city}: {category} ({cnt}次) {match} (预期: {expected})")
            current_city = city

    # 验证"只看不买"用户
    cursor.execute("""
        SELECT COUNT(DISTINCT ub.user_id)
        FROM user_behaviors ub
        WHERE ub.user_id NOT IN (SELECT DISTINCT user_id FROM orders)
    """)
    ghost_count = cursor.fetchone()[0]
    print(f"\n  '只看不买'用户数: {ghost_count} (预期: 10, ID 491-500)")

    # 验证品类分布
    cursor.execute("""
        SELECT category, COUNT(*) FROM products GROUP BY category ORDER BY COUNT(*) DESC
    """)
    print("\n  商品品类分布:")
    for row in cursor.fetchall():
        print(f"    {row[0]}: {row[1]} 件")

    conn.close()

    print(f"\n{'=' * 60}")
    print(f"  [OK] 数据库已生成: {DB_PATH}")
    print(f"  大小: {os.path.getsize(DB_PATH) / 1024:.1f} KB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
