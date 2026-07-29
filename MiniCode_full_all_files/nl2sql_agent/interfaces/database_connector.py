"""
接口 2: IDatabaseConnector —— 数据库连接器

屏蔽不同数据库（SQLite / MySQL / PostgreSQL）的执行差异。
Agent 只依赖接口，不关心底层是什么数据库。

安全设计（难点D）:
  - execute() 前做 SQL 黑名单检查
  - SQLite 以只读模式打开（uri?mode=ro）
  - 返回结果限制最大行数
"""

from abc import ABC, abstractmethod
from nl2sql_agent.schemas import SQLResult, SchemaInfo


class IDatabaseConnector(ABC):
    """
    数据库连接器抽象接口。

    实现类需负责:
      1. 建立/管理数据库连接
      2. 执行 SQL 并返回结构化结果
      3. 提供 Schema 元数据（供 SchemaProvider 使用）
      4. 安全检查（只读、黑名单过滤）
    """

    @abstractmethod
    def execute(self, sql: str, max_rows: int = 50) -> SQLResult:
        """
        执行 SQL 查询并返回结构化结果。

        安全要求（实现类必须遵守）:
          1. 执行前调用 _security_check(sql) 做黑名单 + 白名单校验
          2. 只允许 SELECT 语句
          3. 限制返回行数 ≤ max_rows

        Returns:
            SQLResult: 成功时包含 columns/rows/row_count，
                       失败时包含 error/error_type
        """
        ...

    @abstractmethod
    def get_schema_info(self) -> SchemaInfo:
        """
        从实际数据库中读取完整的 Schema 元数据。

        实现方式（按数据库类型）:
          SQLite:   PRAGMA table_info(...) + sqlite_master
          MySQL:    INFORMATION_SCHEMA.COLUMNS
          PostgreSQL: information_schema.columns + pg_catalog
        """
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        """测试数据库连接是否正常"""
        ...

    @property
    @abstractmethod
    def db_type(self) -> str:
        """数据库类型: 'sqlite' | 'mysql' | 'postgresql'"""
        ...


# ═══════════════════════════════════════════════════════════════
#  SQLiteConnector — Demo 默认实现
# ═══════════════════════════════════════════════════════════════

class SQLiteConnector(IDatabaseConnector):
    """
    SQLite 数据库连接器。

    以只读模式打开数据库，执行前做安全性检查。
    """

    def __init__(self, db_path: str, readonly: bool = True):
        self._db_path = db_path
        self._readonly = readonly
        self._blacklist = self._load_blacklist()

    @staticmethod
    def _load_blacklist() -> list[str]:
        """加载 SQL 黑名单关键字"""
        from nl2sql_agent.config import SQL_BLACKLIST
        return SQL_BLACKLIST

    def _security_check(self, sql: str) -> tuple[bool, str]:
        """
        安全性检查 —— 只拦截真正危险的操作，不限制 SQL 功能。

        核心原则:
          - 防注入/泄露，不是禁用 SQL 功能
          - CTE(WITH) 是标准只读语法，放行
          - 窗口函数、子查询、UNION 全部放行
          - OS 层只读连接作最后防线

        检查项:
          1. 必须以 SELECT 或 WITH 开头（只读查询）
          2. 不能包含黑名单关键字（防注入/泄露/破坏）
        """
        sql_upper = sql.strip().upper()
        # 去掉前导注释
        sql_upper = '\n'.join(
            line for line in sql_upper.split('\n')
            if not line.strip().startswith('--')
        )

        # 允许 SELECT 和 WITH (CTE) —— 两者都是只读的
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return False, "只允许 SELECT 或 WITH (CTE) 查询"

        # 黑名单: 真正危险的操作（防注入 + 防破坏 + 防越权）
        # 不包含 WITH(放行CTE)、不包含 VACUUM/PRAGMA(可讨论)
        import re
        for keyword in self._blacklist:
            if re.search(r'\b' + keyword + r'\b', sql_upper):
                return False, f"禁止的危险操作: {keyword}"

        return True, ""

    def execute(self, sql: str, max_rows: int = 50) -> SQLResult:
        import sqlite3
        import re
        import time
        import urllib.parse

        # ── 安全校验 ──
        ok, err_msg = self._security_check(sql)
        if not ok:
            return SQLResult(
                success=False,
                error=err_msg,
                error_type=None,  # 在 Agent 层映射为 PERMISSION_DENIED
            )

        # ── 执行 SQL ──
        try:
            if self._readonly:
                uri = f"file:{urllib.parse.quote(self._db_path, safe=':/@')}?mode=ro"
                conn = sqlite3.connect(uri, uri=True)
            else:
                conn = sqlite3.connect(self._db_path)

            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            start = time.perf_counter()
            cursor.execute(sql)
            rows = cursor.fetchmany(max_rows + 1)
            elapsed = (time.perf_counter() - start) * 1000

            columns = [d[0] for d in cursor.description] if cursor.description else []
            actual_count = len(rows)
            truncated = actual_count > max_rows
            if truncated:
                rows = rows[:max_rows]

            conn.close()

            return SQLResult(
                success=True,
                columns=columns,
                rows=[tuple(r) for r in rows],
                row_count=actual_count,
                execution_time_ms=elapsed,
            )

        except sqlite3.Error as e:
            error_msg = str(e)
            error_type = self._classify_sqlite_error(error_msg)
            return SQLResult(
                success=False,
                error=error_msg,
                error_type=error_type,
            )

    def _classify_sqlite_error(self, error_msg: str):
        """将 SQLite 错误信息映射为 ErrorType"""
        from nl2sql_agent.config import ErrorType
        msg_lower = error_msg.lower()
        if "no such table" in msg_lower:
            return ErrorType.TABLE_NOT_FOUND
        elif "no such column" in msg_lower:
            return ErrorType.COLUMN_NOT_FOUND
        elif "ambiguous column" in msg_lower:
            return ErrorType.AMBIGUOUS_COLUMN
        elif "syntax error" in msg_lower or "near" in msg_lower:
            return ErrorType.SYNTAX_ERROR
        elif "mismatch" in msg_lower:
            return ErrorType.TYPE_MISMATCH
        else:
            return ErrorType.OTHER

    def get_schema_info(self) -> SchemaInfo:
        import sqlite3
        import urllib.parse

        conn = sqlite3.connect(f"file:{urllib.parse.quote(self._db_path, safe=':/@')}?mode=ro", uri=True) \
            if self._readonly else sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # 获取所有表名（排除系统表和元数据表）
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '\\_%' ESCAPE '\\' "
            "ORDER BY name"
        )
        table_names = [r[0] for r in cursor.fetchall()]

        # 读取中文注释（如果元数据表存在）
        table_descs = {}
        column_descs = {}   # {table_name: {col_name: desc}}
        try:
            cursor.execute(
                "SELECT table_name, description FROM _table_descriptions"
            )
            table_descs = {r[0]: r[1] for r in cursor.fetchall()}
        except sqlite3.OperationalError:
            pass  # 元数据表不存在

        try:
            cursor.execute(
                "SELECT table_name, column_name, description FROM _column_descriptions"
            )
            for row in cursor.fetchall():
                column_descs.setdefault(row[0], {})[row[1]] = row[2]
        except sqlite3.OperationalError:
            pass

        from nl2sql_agent.schemas import TableMeta, ColumnMeta

        tables = []
        for tname in table_names:
            # 获取列信息
            cursor.execute(f"PRAGMA table_info('{tname}')")
            col_rows = cursor.fetchall()

            # 获取外键信息
            cursor.execute(f"PRAGMA foreign_key_list('{tname}')")
            fk_rows = cursor.fetchall()
            fk_cols = {r[3]: f"{r[2]}.{r[4]}" for r in fk_rows}  # col->ref

            # 获取行数
            cursor.execute(f"SELECT COUNT(*) FROM '{tname}'")
            row_count = cursor.fetchone()[0]

            # 该表的列注释
            t_col_descs = column_descs.get(tname, {})

            columns = []
            for c in col_rows:
                col_name = c[1]
                col_type = c[2]
                is_pk = bool(c[5])

                # 获取示例值
                try:
                    cursor.execute(
                        f"SELECT DISTINCT `{col_name}` FROM '{tname}' LIMIT 3"
                    )
                    samples = [str(r[0]) for r in cursor.fetchall()]
                except Exception:
                    samples = []

                columns.append(ColumnMeta(
                    name=col_name,
                    data_type=col_type,
                    description=t_col_descs.get(col_name, ""),
                    is_primary_key=is_pk,
                    is_foreign_key=col_name in fk_cols,
                    foreign_ref=fk_cols.get(col_name, ""),
                    sample_values=samples,
                ))

            tables.append(TableMeta(
                name=tname,
                description=table_descs.get(tname, ""),
                row_count=row_count,
                columns=columns,
            ))

        conn.close()
        return SchemaInfo(db_type="sqlite", tables=tables)

    def test_connection(self) -> bool:
        import sqlite3
        import urllib.parse
        try:
            if self._readonly:
                uri = f"file:{urllib.parse.quote(self._db_path, safe=':/@')}?mode=ro"
                conn = sqlite3.connect(uri, uri=True)
            else:
                conn = sqlite3.connect(self._db_path)
            conn.execute("SELECT 1")
            conn.close()
            return True
        except Exception:
            return False

    @property
    def db_type(self) -> str:
        return "sqlite"
