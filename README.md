# Coding Agent

一个不依赖 LangChain、LangGraph 等 Agent 框架，直接基于模型原生 tool calling 实现的最小编程智能体。项目重点不是堆叠功能，而是完整展示模型调用、工具执行、上下文管理、安全控制和自动验证之间的工作机制。

## 主要能力

- 供应商无关的 `LLMClient` 接口和 OpenAI-compatible Chat Completions 客户端；
- 模型请求超时、有限重试、HTTPS 校验和响应大小限制；
- `Message`、`ToolCall`、`ToolResult` 等核心数据结构；
- 统一工具协议、JSON Schema 导出、参数校验和安全调度；
- `list_files`、`read_file`、`write_file`、`replace_text`、`run_command` 五个工具；
- “模型观察—工具执行—结果回传—模型继续决策”的最小 Agent 循环；
- 上下文预算、完整工具轮次裁剪、最大轮数和重复无进展检测；
- 固定工作目录、敏感文件保护、命令白名单、超时和输出截断；
- 可选的 `--verbose` 脱敏执行轨迹；
- 带明确 Bug 和自动测试的端到端演示项目。

## 环境要求

- Python 3.11 或更高版本；
- 支持 OpenAI-compatible Chat Completions 和原生 tool calling 的模型 API；
- Windows PowerShell、Linux 或 macOS 终端。

项目运行时不依赖第三方 Python 包。

## 安装

在项目根目录创建虚拟环境并执行可编辑安装。

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

安装完成后可以直接使用命令行入口：

```powershell
coding-agent --help
```

也可以通过 Python 模块入口运行：

```powershell
python -m coding_agent --help
```

`main.py` 仅作为源码目录中的兼容开发入口。推荐使用 `coding-agent` 或 `python -m coding_agent`，无需再手工设置 `PYTHONPATH`。

## 配置模型 API

复制示例配置：

```powershell
Copy-Item .env.example .env
```

Linux 或 macOS 使用：

```bash
cp .env.example .env
```

然后编辑项目根目录的 `.env`：

```dotenv
MODEL_API_KEY=你的真实API-Key
MODEL_BASE_URL=https://api.openai.com/v1
MODEL_NAME=你的账户可用的模型ID
MODEL_TIMEOUT=60
MODEL_MAX_RETRIES=2
MODEL_MAX_RESPONSE_BYTES=2000000
AGENT_MAX_TURNS=20
MAX_CONTEXT_CHARS=100000
MAX_NO_PROGRESS_TURNS=3
COMMAND_TIMEOUT=60
MAX_TOOL_OUTPUT=20000
```

| 配置项                       |                        默认值 | 说明                                            |
| ---------------------------- | ----------------------------: | ----------------------------------------------- |
| `MODEL_API_KEY`            |                            无 | 必填；也可以使用`OPENAI_API_KEY`              |
| `MODEL_BASE_URL`           | `https://api.openai.com/v1` | 远程地址必须使用 HTTPS；HTTP 仅允许本机回环地址 |
| `MODEL_NAME`               |                            无 | 必填；模型必须支持 tool calling                 |
| `MODEL_TIMEOUT`            |                        `60` | 单次模型请求超时秒数，必须大于 0                |
| `MODEL_MAX_RETRIES`        |                         `2` | 瞬时错误的额外重试次数，范围 0～5               |
| `MODEL_MAX_RESPONSE_BYTES` |                   `2000000` | 模型响应大小上限，范围 1024～10000000 字节      |
| `AGENT_MAX_TURNS`          |                        `20` | 单次任务允许的最大模型轮数                      |
| `MAX_CONTEXT_CHARS`        |                    `100000` | 发送给模型的上下文字符预算，最小 1000           |
| `MAX_NO_PROGRESS_TURNS`    |                         `3` | 相同工具调用和结果连续重复多少轮后停止，最小 2  |
| `COMMAND_TIMEOUT`          |                        `60` | 命令默认及最大超时，范围 1～60 秒               |
| `MAX_TOOL_OUTPUT`          |                     `20000` | 单个工具返回给模型的字符上限，最小 100          |

进程环境变量优先于 `.env`。真实 API Key 只能保存在进程环境变量或未入库的 `.env` 中；不要写入源码、README、`.env.example` 或 Git 历史。

## 快速开始

让 Agent 自行观察一个不知道文件名的项目并总结用途：

```powershell
coding-agent --workspace examples/demo_project "请自行查看这个项目并总结用途"
```

显示脱敏执行轨迹：

```powershell
coding-agent --verbose --workspace examples/demo_project "请自行查看这个项目并总结用途"
```

不传提示词时，程序会在终端中交互询问：

```powershell
coding-agent --workspace examples/demo_project
```

最终回答写入 stdout；`--verbose` 轨迹写入 stderr，只显示模型轮次、工具名称、脱敏参数和结果摘要，不显示 API Key、完整文件内容或编辑文本。

## 命令行参数

```text
usage: coding-agent [-h] [--workspace WORKSPACE] [--system SYSTEM] [--verbose] [prompt]
```

- `prompt`：要交给 Agent 的任务；省略时交互输入；
- `--workspace`：工具能够访问的工作目录，默认为当前目录；
- `--system`：覆盖本次运行的系统提示词；
- `--verbose`：显示脱敏后的模型与工具执行轨迹。

## 工作流程

一次典型任务按以下顺序执行：

1. 模型根据用户任务决定直接回答还是调用工具；
2. Agent 校验工具名称、参数和调用 ID；
3. 工具只在指定工作目录内执行；
4. 工具结果以 `tool` message 返回模型；
5. 模型继续观察、修改并运行测试验证；
6. 模型给出最终答案，或由轮数、上下文和无进展策略安全终止。

完整审计历史保留在 `AgentRunResult.history` 中；发送给模型的上下文超出预算时，只裁剪旧的完整 assistant/tool 交换，避免产生孤立工具消息。

## 工具与安全边界

| 工具             | 用途               | 主要限制                                         |
| ---------------- | ------------------ | ------------------------------------------------ |
| `list_files`   | 查看目录结构       | 限制递归深度、条目数和输出大小，隐藏敏感路径     |
| `read_file`    | 读取 UTF-8 文本    | 禁止目录穿越、工作区逃逸、敏感文件和二进制文件   |
| `write_file`   | 创建新文件         | 默认禁止覆盖，父目录必须已存在                   |
| `replace_text` | 精确修改文件       | 旧文本必须恰好出现一次，使用原子替换             |
| `run_command`  | 运行测试和只读检查 | 固定工作目录、命令白名单、超时、退出码和双流截断 |

`run_command` 不提供任意 Shell：

- Python 只允许通过 `-m` 运行 `unittest`、`pytest` 或 `compileall`；
- 禁止直接执行 `.py` 文件、内联代码和标准输入代码；
- Git 仅允许只读子命令，并且工作区根目录自身必须包含 `.git`；
- 子进程会清理常见密钥和凭据环境变量；
- stdout 与 stderr 都会受统一输出预算限制。

这些约束用于降低误操作风险，不等价于操作系统级沙箱；请仍然为 Agent 指定独立、可恢复的工作目录。

## 端到端演示

`examples/buggy_shipping` 是一个小型运费计算项目，包含明确的边界 Bug 和 8 项自动测试。它用于验证 Agent 是否能完成“观察—修改—验证”闭环：

```powershell
coding-agent --verbose --workspace examples/buggy_shipping "请检查这个项目，定位并修复 Bug，不要修改测试，最后运行完整测试验证"
```

期望行为包括：

1. 先列出项目文件；
2. 阅读需求、实现和测试；
3. 运行测试观察失败；
4. 使用 `replace_text` 精确修复；
5. 再次运行完整测试；
6. 根据真实退出码和测试输出汇报结果。

## 运行测试

完成可编辑安装后运行：

```powershell
python -m unittest discover -s tests -v
```

如果尚未安装，也可以从源码目录临时运行：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

当前测试覆盖核心数据结构、配置、模型客户端、工具注册、安全文件操作、受限命令、上下文裁剪、无进展检测、CLI、安装入口和完整工作流。

## 项目结构

```text
.
├── main.py                         # 源码开发兼容入口
├── pyproject.toml                  # 包元数据与 coding-agent 命令入口
├── src/coding_agent/
│   ├── __main__.py                 # python -m coding_agent 入口
│   ├── agent.py                    # AgentRunner 与上下文控制
│   ├── cli.py                      # 命令行解析、工具装配与轨迹脱敏
│   ├── config.py                   # .env 和运行配置校验
│   ├── messages.py                 # Message 与 ToolCall
│   ├── llm/                        # 模型统一接口与兼容客户端
│   └── tools/                      # 工具协议、注册表和具体工具
├── examples/
│   ├── demo_project/
│   └── buggy_shipping/
└── tests/
```
