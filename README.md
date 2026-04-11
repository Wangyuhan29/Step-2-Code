# GitHub Rust 数据处理与情感分析工具

本项目是 [Step-1-Code](https://github.com/Wangyuhan29/Step-1-Code) 的后续项目。  
Step-1-Code 负责从 GitHub API 采集 Rust 社区数据并存入 `github_rust_data` MySQL 数据库；  
本项目（Step-2-Code）负责对采集到的数据进行**手动 SQL 筛选**和**情感分类**。

## 功能特性

### 第一步：手动 SQL 筛选
- 你自行编写 `SELECT` 语句从原始表筛选需要分析的数据。
- 查询结果至少需要包含 `id`（或别名 `text_id`）和可分析文本列。
- 文本列优先读取 `text`，若无则自动拼接 `title/description/body`。

### 第二步：情感分类（基于 DeepSeek API）
使用 DeepSeek 大语言模型对预处理后的文本进行细粒度情感分析。
当前 Prompt 已调整为**方面级情感分析（ABSA）**：
- 一条文本可包含多个 aspect（通常不超过 3 个）
- 每个 aspect 输出 `sentiment`（`positive` / `negative` / `neutral`）
- 同时输出离散强度 `score`（`2/1/0/-1/-2`）

可选 aspect 列表：
- Language：`ownership`、`type_system`、`safety`、`performance`
- Experience：`learning_curve`、`compile_time`、`error_message`、`debugging`
- Engineering：`maintainability`、`readability`、`extensibility`、`api_design`
- Ecosystem：`package_manager`、`libraries`、`framework_support`、`community`

Prompt 设计要素：
- 详细的 aspect 定义与判断维度说明
- 针对 GitHub 技术社区的领域适配说明
- Few-shot 示例引导（输入/输出成对）
- 严格 JSON 输出约束（禁止 Markdown/表格/代码块）
- 响应侧仅接受 JSON，不再兼容或清洗制表/Markdown 格式输出

模型输出格式（严格 JSON）：

```json
{
  "annotations": [
    {
      "aspect": "learning_curve",
      "sentiment": "negative",
      "score": -2
    }
  ]
}
```

### 数据输出
情感分析结果输出到 JSON 文件（默认 `sentiment_output.json`），每条记录仅包含：
- `text_id`：原始表 `id`
- `annotations`：模型输出的 aspect 标注结果

## 环境要求

- Python 3.8+
- MySQL 5.7+ 或 MariaDB 10.2+（已有 `github_rust_data` 数据库）
- DeepSeek API Key（从 [platform.deepseek.com](https://platform.deepseek.com) 获取）

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/Wangyuhan29/Step-2-Code.git
cd Step-2-Code
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如遇到认证问题，可加装 cryptography：

```bash
pip install cryptography
```

### 3. 配置文件

```bash
cp config.example.ini config.ini
```

编辑 `config.ini`：

```ini
[deepseek]
api_key = your_deepseek_api_key_here
base_url = https://api.deepseek.com
model = deepseek-chat

[mysql]
host = localhost
port = 3306
user = root
password = your_mysql_password
database = github_rust_data

[processor]
# 允许的语言（langdetect 语言代码，逗号分隔）
allowed_languages = en,zh-cn,zh-tw
min_text_length = 10
dedup_strategy = exact
save_to_db = true

[sentiment]
batch_size = 10
request_delay = 1.0
max_retries = 3
output_json_file = sentiment_output.json
```

### 4. 使用环境变量（可选）

```bash
export DEEPSEEK_API_KEY=your_api_key
export MYSQL_PASSWORD=your_password
```

## 使用方法

### 手动 SQL 筛选并执行情感分析

SQL命令写入QUERY.sql

### 终端执行：

```bash
python .\main.py --config .\config.ini --sql-file .\QUERY.sql --output-json .\sentiment_output.json
```


## 项目结构

```
Step-2-Code/
├── README.md              # 项目文档
├── requirements.txt       # Python 依赖
├── config.example.ini     # 配置文件模板
├── config.py              # 配置管理模块
├── database.py            # 数据库操作模块
├── processor.py           # 数据预处理模块
├── sentiment_analyzer.py  # DeepSeek 情感分析模块
└── main.py                # 程序入口
```

## SQL 筛选示例

```sql
-- issues：拼接 title/description 作为待分析文本
SELECT id, CONCAT_WS('\n\n', title, description) AS text
FROM issues
WHERE created_at >= '2025-01-01';

-- issue_comments：直接用 body
SELECT id, body AS text
FROM issue_comments
WHERE body IS NOT NULL AND body != '';
```

`sentiment_output.json` 示例结构：

```json
[
  {
    "text_id": 123,
    "annotations": [
      {
        "aspect": "learning_curve",
        "sentiment": "negative",
        "score": -2
      }
    ]
  }
]
```

## 注意事项

1. **运行顺序**：请先运行 Step-1-Code 采集数据，再运行本项目分析数据。
2. **API 费用**：DeepSeek API 按调用量计费，建议先用少量数据测试。
3. **速率限制**：`request_delay` 参数控制请求间隔，避免触发 API 速率限制。
4. **SQL 结果要求**：`--sql` 必须是 `SELECT` 语句，并返回 `id`（或 `text_id`）与可分析文本列。

## 许可证

本项目使用 MIT 许可证。