"""Peer registration adapter for the existing RAG implementation."""


AGENT_SPEC = {
    "id": "rag",
    "name": "RAG Agent",
    "icon": "📚",
    "description": "从本地知识库和上传文档中检索相关内容并回答问题",
    "keywords": ["知识库", "文档", "资料", "检索", "总结"],
    "intents": ["knowledge_retrieval", "document_question"],
    "examples": ["从知识库查找部署说明", "根据上传文档回答问题"],
    "factory": "agents.rag_agent:create_agent",
    "config_namespace": "rag",
    "input_schema": {
        "required": ["instruction"],
        "properties": {},
    },
    "output_schema": {
        "exports": ["answer", "summary", "data.structured"],
    },
    "capabilities": {
        "needs_history": True,
        "needs_files": True,
        "needs_long_term": False,
        "max_context_chars": 6000,
        "side_effects": "none",
        "auto_chainable": True,
        "needs_credentials": [],
    },
}

AGENT_META = AGENT_SPEC


def create_agent():
    from wrappers.rag_wrapper import RAGWrapper

    return RAGWrapper()


__all__ = ["AGENT_META", "AGENT_SPEC", "create_agent"]

