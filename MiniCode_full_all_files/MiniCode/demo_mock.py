"""Mock Agents — NL2SQL / RAG 加载失败时的降级方案"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MockNL2SQL:
    """模拟 NL2SQL Agent"""

    DATA = {
        "北京": [
            ("故宫文创书签", 1520, 45600),
            ("景泰蓝手镯", 1200, 72000),
            ("青花瓷U盘", 980, 29400),
            ("丝绸围巾", 860, 43000),
            ("长城模型", 740, 37000),
        ],
        "上海": [
            ("老上海雪花膏", 2100, 31500),
            ("丝绸围巾", 1850, 92500),
            ("故宫文创书签", 1400, 42000),
        ],
        "default": [
            ("故宫文创书签", 3400, 102000),
            ("丝绸围巾", 2800, 140000),
            ("景泰蓝手镯", 2200, 132000),
            ("青花瓷U盘", 1800, 54000),
            ("长城模型", 1500, 75000),
        ],
    }

    def run(self, question: str) -> dict:
        city = "default"
        for c in ["北京", "上海", "广州", "深圳"]:
            if c in question:
                city = c
                break
        data = self.DATA.get(city, self.DATA["default"])
        top = data[0]
        summary = f"{city}销量最高: {top[0]} ({top[1]}件, {top[2]}元)"
        return {
            "status": "success", "answer": summary,
            "data": {"rows": data, "city": city, "sql": "-- mock"},
            "summary": summary,
        }


class MockRAG:
    """模拟 RAG Agent"""

    def run(self, question: str) -> dict:
        return {
            "status": "success",
            "answer": f"[Mock RAG] 检索到 3 篇相关文档，覆盖: {question[:40]}...",
            "data": {"citations": ["doc_001", "doc_003"], "chunks": 3},
            "summary": "检索到 3 篇相关文档",
        }
