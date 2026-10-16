"""Skill Loader —— 自动发现、校验、注册 skills/ 目录下的 Python 模块

每个 skill 文件必须包含:
  META = {"name": "...", "description": "...", "keywords": [...], "icon": "..."}
  def run(query: str, context: dict = None) -> dict
"""
import os, sys, json, time, importlib.util, inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class _SkillDef:
    module_name: str
    meta: dict
    run_func: Callable
    source_path: str
    loaded_at: float = field(default_factory=time.time)


class SkillLoader:
    """Skill 发现和注册"""

    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir
        self.skills: Dict[str, _SkillDef] = {}
        self.errors: List[str] = []
        os.makedirs(self.skills_dir, exist_ok=True)

    def discover(self) -> int:
        """扫描 skills/*.py，加载所有有效 skill，返回成功数量"""
        self.skills.clear()
        self.errors.clear()

        if not os.path.isdir(self.skills_dir):
            return 0

        py_files = [f for f in os.listdir(self.skills_dir)
                    if f.endswith(".py") and not f.startswith("_") and f != "__init__.py"]

        for fname in sorted(py_files):
            path = os.path.join(self.skills_dir, fname)
            mod_name = fname[:-3]  # strip .py
            skill_id = f"skill:{mod_name}"

            try:
                spec = importlib.util.spec_from_file_location(f"skill_{mod_name}", path)
                if spec is None or spec.loader is None:
                    self.errors.append(f"{fname}: 无法加载模块")
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                self.errors.append(f"{fname}: import 失败 - {e}")
                continue

            # 校验 META
            if not hasattr(module, "META") or not isinstance(module.META, dict):
                self.errors.append(f"{fname}: 缺少 META dict")
                continue
            meta = module.META
            if not meta.get("name"):
                self.errors.append(f"{fname}: META.name 为空")
                continue
            if not meta.get("description"):
                self.errors.append(f"{fname}: META.description 为空")
                continue

            # 校验 run()
            if not hasattr(module, "run") or not callable(module.run):
                self.errors.append(f"{fname}: 缺少 run() 函数")
                continue

            self.skills[skill_id] = _SkillDef(
                module_name=mod_name,
                meta=meta,
                run_func=module.run,
                source_path=path,
            )
            print(f"  [Skill] 已加载: {skill_id} ({meta['name']})")

        return len(self.skills)

    def get_skill(self, skill_id: str) -> Optional[Callable]:
        """根据 skill_id（如 skill:weather）获取 run() 函数"""
        sd = self.skills.get(skill_id)
        return sd.run_func if sd else None

    def list_as_agent_configs(self) -> dict:
        """返回兼容 ROUTER_AGENTS 格式的配置字典"""
        configs = {}
        for sid, sd in self.skills.items():
            configs[sid] = {
                "name": sd.meta.get("name", sid),
                "icon": sd.meta.get("icon", "🧩"),
                "description": sd.meta.get("description", ""),
                "keywords": sd.meta.get("keywords", []),
                "source": "skill",
            }
        return configs

    def register_in_loader_registry(self, loaders: dict):
        """将 skill factory 注册到 AGENT_LOADERS"""
        for sid, sd in self.skills.items():
            run_func = sd.run_func
            loaders[sid] = lambda rf=run_func: _SkillWrapper(rf, sid)

    def register_tools_in_registry(self, registry):
        """如果 skill 导出 TOOLS 列表，注册到 ToolRegistry"""
        # NOTE: 此方法在 skill 文件内容变化时不会自动感知——模块已被系统缓存
        # (sys.modules)，即使磁盘上的 .py 文件已更新，返回的仍是旧模块。
        # 热更新需先 del sys.modules[mod_key] 再重新 load 或重启 web server。
        from mini_thalamus.tool_registry import ToolDef
        for sid, sd in self.skills.items():
            path = sd.source_path
            mod_key = f"skill_{sd.module_name}"
            # 如果已经加载过，复用缓存的模块（避免重复执行模块级代码）
            if mod_key in sys.modules:
                module = sys.modules[mod_key]
            else:
                try:
                    spec = importlib.util.spec_from_file_location(mod_key, path)
                    module = importlib.util.module_from_spec(spec)
                    if spec.loader:
                        spec.loader.exec_module(module)
                except Exception:
                    continue
            try:
                tools = getattr(module, "TOOLS", [])
                for t in tools:
                    if isinstance(t, dict):
                        registry.register(ToolDef(
                            name=f"{sid}.{t['name']}",
                            description=t.get("description", ""),
                            parameters=t.get("parameters", {"type": "object", "properties": {}}),
                            handler=t["handler"],
                            agent=sid,
                            source="skill",
                        ))
            except Exception:
                pass


class _SkillWrapper:
    """包装 skill 的 run() 函数为标准 Agent 接口"""
    def __init__(self, run_func, skill_id: str):
        self._run = run_func
        self.skill_id = skill_id

    def run(self, query: str, context: dict = None) -> dict:
        try:
            result = self._run(query, context)
            if not isinstance(result, dict):
                return {"status": "success", "answer": str(result),
                        "data": {}, "summary": str(result)[:300]}
            return result
        except Exception as e:
            return {"status": "error",
                    "answer": f"Skill 执行失败: {e}",
                    "data": {}, "summary": str(e)}
