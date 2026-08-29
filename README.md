# Coding Agent

一个不依赖 Agent 框架、以模型原生 tool calling 为基础的最小编程智能体。

## 当前状态

项目已经完成：

- `Message`、`ToolCall`、`ToolResult` 三个核心数据结构；
- `AppConfig` 环境与本地配置加载；
- 供应商无关的 `LLMClient` 接口；
- OpenAI-compatible Chat Completions 客户端；
- 统一 `Tool` 协议和 `ToolResult` 执行结果；
- `ToolRegistry` 注册、Schema 导出、参数校验和安全调度；
- 支持递归深度和输出限制的 `ListFilesTool`；
- 受工作目录限制、支持行范围和输出截断的 `ReadFileTool`；
- 只允许创建新文件、禁止覆盖的 `WriteFileTool`；
- 仅在旧文本恰好出现一次时执行原子更新的 `ReplaceTextTool`；
- 固定工作目录、限制命令、支持超时和有界双流输出的 `RunCommandTool`；
- `AgentRunner` 模型—工具—模型循环、最大轮数和调用 ID 检查；
- 支持 `--workspace` 的最小 Agent CLI。

## 环境要求

- Python 3.11+

## 配置模型 API

项目启动时会自动读取项目根目录的 `.env`。打开本地 `.env`，填写：

```dotenv
MODEL_API_KEY=你的真实API Key
MODEL_BASE_URL=https://api.openai.com/v1
MODEL_NAME=你的账户可用的模型ID
COMMAND_TIMEOUT=60
```

`COMMAND_TIMEOUT` 必须是 1～60 的整数秒，同时作为单次命令的默认超时和最大超时。

`.env` 已被 `.gitignore` 排除，不应提交到仓库。`.env.example` 只保存变量名和示例值，可以安全提交。

系统环境变量的优先级高于 `.env`，因此部署或临时切换模型时仍可以覆盖本地配置。

## 运行 Agent

在项目根目录运行：

```powershell
$env:PYTHONPATH = "src"
python main.py --workspace examples/demo_project "请自行查看这个项目并总结用途"
```

增加 `--verbose` 可以显示模型轮次、工具名称、安全参数和结果摘要。轨迹写入
stderr，不显示工具输出正文或模型 API Key：

```powershell
python main.py --verbose --workspace examples/demo_project "请自行查看这个项目并总结用途"
```

也可以不传提示词，在终端交互输入：

```powershell
$env:PYTHONPATH = "src"
python main.py
```

## 运行测试

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## 端到端演示项目

`examples/buggy_shipping` 包含一个运费分档边界 Bug 和 8 项自动测试。修复前有
2 项测试失败，适合演示 Agent 自主查看需求、定位代码、精确修改并运行测试验证：

```powershell
python main.py --verbose --workspace examples/buggy_shipping "请检查这个项目，定位并修复 Bug，不要修改测试，最后运行完整测试验证"
```

## 项目结构

```text
src/coding_agent/
├── agent.py
├── cli.py
├── config.py
├── messages.py
├── llm/
└── tools/
```

真实 API Key 只能通过环境变量或未入库的 `.env` 文件提供，禁止写入源码、README、`.env.example` 或 Git 历史。
