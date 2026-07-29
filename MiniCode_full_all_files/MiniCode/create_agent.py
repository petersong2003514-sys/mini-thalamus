#!/usr/bin/env python3
"""Create a hot-pluggable Agent module from ``agents/_template.py``."""

import argparse
import os
import re
import sys


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MINICODE_ROOT = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(MINICODE_ROOT, "agents")
TEMPLATE_PATH = os.path.join(AGENTS_DIR, "_template.py")


def snake_case(name: str) -> str:
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value.lower()


def generate_agent(
    class_name: str,
    description: str,
    icon: str = "🔧",
    keywords: list = None,
    dry_run: bool = False,
    add_router: bool = True,
):
    """Generate one Agent file; AgentRegistry performs registration."""
    del add_router  # retained for command-line compatibility
    keywords = keywords or re.findall(r"[\w一-鿿]{2,}", description)[:5]
    keywords = keywords or [class_name]

    file_name = snake_case(class_name)
    agent_id = file_name.removesuffix("_agent")
    agent_path = os.path.join(AGENTS_DIR, f"{file_name}.py")
    if os.path.exists(agent_path):
        print(f"  ✗ Agent 文件已存在: {agent_path}")
        return False

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as handle:
        content = handle.read()

    content = content.replace("MyAgent", class_name)
    content = content.replace('"name": "新 Agent"', f'"name": "{class_name}"')
    content = content.replace('"""新 Agent 实现"""', f'"""{class_name} 实现"""')
    content = content.replace(
        "这个 Agent 做什么——会显示在侧边栏和 Router 描述中",
        description,
    )
    content = content.replace('"id": "my"', f'"id": "{agent_id}"')
    content = content.replace(
        '"factory": "agents.my_agent:create_agent"',
        f'"factory": "agents.{file_name}:create_agent"',
    )
    content = content.replace(
        '"config_namespace": "my"',
        f'"config_namespace": "{agent_id}"',
    )
    content = content.replace(
        '"关键词1", "关键词2"',
        ", ".join(f'"{keyword}"' for keyword in keywords),
    )
    content = content.replace('"icon": "🔧"', f'"icon": "{icon}"')

    if dry_run:
        print(f"  [DRY-RUN] 将创建: {agent_path}")
        print("\n".join(content.splitlines()[:45]))
        return True

    with open(agent_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    print(f"  ✓ 已创建: agents/{file_name}.py")
    print(f"  ✓ AgentRegistry 将自动发现 id={agent_id}")
    print("  重启服务或调用 Registry.refresh/discover 后生效")
    return True


def main():
    parser = argparse.ArgumentParser(description="创建可热插拔的 Agent")
    parser.add_argument("class_name", help="CamelCase 类名，例如 VisionAgent")
    parser.add_argument("description", help="Agent 能力描述")
    parser.add_argument("--icon", default="🔧")
    parser.add_argument("--keywords", help="逗号分隔的关键词")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-router",
        action="store_true",
        help="兼容参数；Registry 架构下无需手工修改 Router",
    )
    args = parser.parse_args()

    if not re.match(r"^[A-Z][a-zA-Z0-9]*Agent$", args.class_name):
        parser.error("类名必须是以 Agent 结尾的 CamelCase 名称")

    keywords = None
    if args.keywords:
        keywords = [item.strip() for item in args.keywords.split(",") if item.strip()]

    success = generate_agent(
        class_name=args.class_name,
        description=args.description,
        icon=args.icon,
        keywords=keywords,
        dry_run=args.dry_run,
        add_router=not args.no_router,
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
