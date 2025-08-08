# 智能文本转语音自动化流程 (Text-to-Speech Automation)

这是一个功能强大的 Python 应用程序，旨在将长篇文本（如小说、报告等）自动化地转换为高质量的音频文件。它通过一个健壮的、可续传的工作流，将文本智能地分割成语义连贯的块，然后逐一调用 TTS (Text-to-Speech) 服务生成语音，最终将音频片段合并成完整的章节音频。

## ✨ 主要功能

- **自动化工作流**: 只需提供结构化的输入，即可自动完成从文本到语音的全部转换流程。
- **智能混合分块 (Hybrid Splitting)**:
    - **语义优先**: 首先使用 Embedding 模型分析句子间的语义关系，在话题自然过渡处进行分割，保证内容的连贯性。
    - **大小约束**: 对语义分割后的块应用最大长度限制 (`CHUNK_SIZE`)，对过长的块进行细分，确保每个块的大小均匀可控，以适应 TTS 服务和网关的超时限制。
- **断点续传**:
    - **阶段性执行**: 整个流程分为“分割”和“TTS”两个主要阶段。
    - **精确进度记录**: 进度被实时记录在 `metadata.json` 文件中，精确到正在处理的章节和文本块。
    - **自动恢复**: 如果程序意外中断，下次启动时会自动读取进度并从上次中断的地方无缝继续，无需从头开始。
- **健壮的错误处理**:
    - **自动重试**: 在调用 TTS 服务失败时，会自动进行最多3次重试。
    - **失败中止**: 如果一个块在多次重试后仍然失败，程序会中止当前任务，以防止产生不完整的音频文件。
- **高度可配置**:
    - **环境变量**: 所有关键参数（如 API 地址、Key、超时时间、分割阈值等）均通过 `.env` 文件进行配置，实现了配置与代码的分离，安全且灵活。
- **详细的日志系统**: 记录每一步的操作，包括详细的 HTTP 请求与响应日志（需开启 DEBUG 模式），便于追踪和调试。

## 📂 项目结构



text2speech_project/
├── results/ # 输出目录
│ ├── metadata/ # 存放任务元数据 JSON 文件 (输入)
│ ├── splitdata/ # 存放原始章节文本文件 (输入)
│ ├── voicedata/ # 存放最终生成的章节音频 .wav 文件 (输出)
│ └── vectortemp/ # 存放临时的文本分块 .txt 文件 (中间产物)
│
├── src/
│ └── text2speech/ # 应用源代码包
│ ├── init.py # 应用入口和日志配置
│ ├── config.py # 配置加载模块
│ ├── main.py # 任务编排器
│ ├── processor.py # 核心处理逻辑 (分割, TTS, 合并)
│ └── text_utils.py # 文本清理工具
│
├── .env.example # 环境变量示例文件
├── pyproject.toml # 项目定义和依赖管理
└── ...
## 🚀 安装与设置

1.  **克隆仓库**
    ```bash
    git clone <your-repository-url>
    cd text2speech_project
    ```

2.  **安装依赖**
    本项目使用 `uv` 或 `pip` 进行包管理。所有依赖项都定义在 `pyproject.toml` 文件中。
    ```bash
    # 推荐使用 uv
    uv pip install -e .

    # 或者使用 pip
    pip install -e .
    ```-e` 参数表示以“可编辑”模式安装，您对代码的任何修改都会立即生效，无需重新安装。

3.  **配置环境变量**
    - 将 `.env.example` 文件复制一份并重命名为 `.env`。
      ```bash
      cp .env.example .env
      ```
    - 编辑新的 `.env` 文件，填入您自己的配置，特别是 `EMBEDDING_API_KEY` 和服务地址。`.env` 文件中的变量将在程序启动时被自动加载。

## 📖 使用方法

1.  **准备输入数据**
    - **元数据**: 在 `results/metadata/` 目录下，为每个任务创建一个 `{taskId}-metadata.json` 文件。
    - **章节文本**: 在 `results/splitdata/{taskId}/` 目录下，存放对应的章节纯文本文件（如 `1.txt`, `2.txt` 等）。

2.  **运行应用**
    在项目根目录下，执行以下命令：
    ```bash
    uv run text2speech
    ```
    程序会自动扫描 `results/metadata` 目录，并开始处理所有待处理的任务。

3.  **查看输出**
    - **临时分块**: 分割后的文本块会临时存储在 `results/vectortemp/{taskId}/` 目录下。
    - **最终音频**: 生成的章节音频文件会保存在 `results/voicedata/{taskId}/` 目录下。
    - **进度更新**: `metadata.json` 文件会被实时更新以反映当前的处理进度。

## ⚙️ 配置详解

您可以在 `.env` 文件中修改以下环境变量：

| 变量名                   | 描述                                                               | 默认值                         |
| ------------------------ | ------------------------------------------------------------------ |-----------------------------|
| `CHUNK_SIZE`             | 文本块的最大字符数限制。                                           | `4000`                      |
| `CHUNK_OVERLAP`          | 当一个大块被细分时，相邻小块之间的重叠字符数。                     | `200`                       |
| `TTS_BASE_URL`           | 您的 TTS 服务的基础 URL 地址。                                     | `http://127.0.0.1:8081`     |
| `TTS_ENDPOINT`           | TTS 服务的具体端点。                                               | `/tts/fast`                 |
| `TTS_REQUEST_TIMEOUT`    | 调用 TTS 服务的客户端超时时间（秒）。                              | `600`                       |
| `EMBEDDING_API_KEY`      | 您的 Embedding 服务的 API Key。                                    | `123456`                    |
| `EMBEDDING_BASE_URL`     | 您的 Embedding 服务的基础 URL 地址。                               | `http://127.0.0.1:8080/v1`  |
| `EMBEDDING_MODEL_NAME`   | 您使用的 Embedding 模型名称。                                      | `Qwen3-Embedding-8B-Q4_K_M` |
| `SEMANTIC_SPLIT_THRESHOLD` | 语义分割的阈值 (0.0-1.0)。值越高，分割出的块越短。               | `0.3`                       |
| `LOG_LEVEL`              | 应用的日志级别 (DEBUG, INFO, WARNING, ERROR)。                     | `INFO`                      |

## 🛠️ 工作流程简述

程序严格按照以下顺序，通过更新 `metadata.json` 文件来驱动和恢复流程：

1.  **初始化**: 读取 `metadata.json`，检查 `progress.current_phase` 字段以确定从何处开始。
2.  **分割阶段 (`splitting`)**:
    - 遍历所有章节，对尚未处理的章节进行混合分割。
    - 将分割后的文本块保存到 `vectortemp` 目录。
    - 每完成一个章节的分割，就更新一次 `metadata.json`。
    - 此阶段全部完成后，将主状态更新为 `tts`。
3.  **TTS 阶段 (`tts`)**:
    - 遍历所有文本块，对尚未处理的块进行 TTS 转换。
    - 每成功转换一个块，就更新一次 `metadata.json` 中该块的状态。
    - 如果转换失败，则进行重试。若重试全部失败，则中止任务。
    - 当一个章节的所有块都转换成功后，合并音频，清理临时文件，并更新 `metadata.json`。
4.  **完成 (`completed`)**: 所有章节都成功生成音频后，将主状态更新为 `completed`，并记录最终信息。

## 📄 许可证

本项目采用 [MIT](LICENSE) 许可证。



