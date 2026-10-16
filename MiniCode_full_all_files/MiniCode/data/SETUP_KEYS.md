# 一键启用的外部能力

## 1. WebSearch 稳定后端（推荐 Tavily）
1. 注册 https://tavily.com 并创建 API Key。
2. 启动服务前设置：
```powershell
$env:SHUSHU_SEARCH_ENGINE="tavily"
$env:SHUSHU_SEARCH_API_KEY="你的key"
```
或使用 Serper：`SHUSHU_SEARCH_ENGINE=serper` + Serper key；
或本地 SearXNG：`SHUSHU_SEARCH_ENGINE=searxng` + `SHUSHU_SEARXNG_URL=http://localhost:8080`。
不配置时仍走 Bing 兜底。

## 2. api_caller 规格
`data/api_specs/*.json`，格式见 `example.json`。已内置免 key 的：
- `beijing_weather.json`：北京实时天气（Open-Meteo）
- `public_ip.json`：公网 IP（ipify）
- `example.json`：GitHub 用户（octocat）

## 3. MCP
`data/.mcp.json` 已启用官方 filesystem server，允许操作 `<your-workspace>`：
```json
{ "mcpServers": { "filesystem": { "type": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "<your-workspace>"], "env": {} } } }
```
重启服务后 `/api/mcp/status` 应显示可用。要接其他 MCP server，按同格式追加。

## 4. OCR 模型目录
以后新增模型前设置：
```powershell
$env:EASYOCR_MODULE_PATH="D:/easyocr_models"
```

## 5. WebSearch 自动降级

默认顺序：Tavily → Serper → Bocha → SearXNG → Bing。
- `SHUSHU_SEARCH_ENGINE` 可写多个，如 `tavily,serper`；不写也按默认顺序尝试有 key 的 provider。
- 每个 provider 可用独立 key：`TAVILY_API_KEY`、`SERPER_API_KEY`、`BOCHA_API_KEY`；没有独立 key 时用 `SHUSHU_SEARCH_API_KEY`。
- Tavily 额度用尽或报错时会自动尝试下一个，全部失败再走 Bing。