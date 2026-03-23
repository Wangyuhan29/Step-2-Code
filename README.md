# GitHub Rust 数据处理与情感分析工具

本项目是 [Step-1-Code](https://github.com/Wangyuhan29/Step-1-Code) 的后续项目。  
Step-1-Code 负责从 GitHub API 采集 Rust 社区数据并存入 `github_rust_data` MySQL 数据库；  
本项目（Step-2-Code）负责对采集到的数据进行**预处理**和**情感分类**。

## 功能特性

### 第一步：结构解析与数据预处理
- **噪声剥离**：去除代码块（\`\`\`...\`\`\` 和行内代码）、HTML 标签与实体、URL、日志行、堆栈跟踪、Diff 行、Markdown 引用块等
- **去重**：基于 SHA-256 内容哈希的精确去重，避免重复分析相同内容
- **语言过滤**：使用 `langdetect` 自动检测语言，只保留指定语言（默认保留英文和中文）的文本
- **文本规范化**：Unicode NFC 标准化、统一换行符、合并多余空行和空格

### 第二步：情感分类（基于 DeepSeek API）
使用 DeepSeek 大语言模型对预处理后的文本进行四分类情感分析：

| 标签 | 含义 |
|------|------|
| `positive` | 正面情感（赞扬、感谢、满意、积极期待等）|
| `negative` | 负面情感（批评、抱怨、沮丧、不满等）|
| `neutral`  | 中性/客观（技术描述、功能说明等）|
| `mixed`    | 混合情感（同时包含明显正面和负面情感）|

Prompt 设计要素：
- 详细的标签定义与判断维度说明
- 针对 GitHub 技术社区的领域适配说明
- 四组 Few-shot 示例（每个标签各一例）
- 严格的 JSON 输出格式约束（`label`、`confidence`、`reasoning`）

### 数据存储
处理结果写入 `github_rust_data` 数据库的两张新表：
- `processed_texts`：预处理结果（原始文本、清洗后文本、检测语言、是否重复）
- `sentiment_results`：情感分类结果（标签、置信度、分析理由、使用模型）

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
```

### 4. 使用环境变量（可选）

```bash
export DEEPSEEK_API_KEY=your_api_key
export MYSQL_PASSWORD=your_password
```

## 使用方法

### 全流程（预处理 + 情感分析）

```bash
python main.py
```

### 仅执行预处理

```bash
python main.py --step preprocess
```

### 仅执行情感分析

```bash
python main.py --step analyze
```

### 通过命令行传入 API Key

```bash
python main.py --api-key your_deepseek_api_key
```

### 使用自定义配置文件

```bash
python main.py --config /path/to/your/config.ini
```

## 数据库结构

### processed_texts 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 主键（自增）|
| source_type | VARCHAR(20) | 来源类型：issue / pr / issue_comment / pr_comment |
| source_id | BIGINT | 原始记录 ID |
| raw_text | MEDIUMTEXT | 原始文本 |
| processed_text | MEDIUMTEXT | 预处理后的文本 |
| language | VARCHAR(20) | 检测到的语言代码 |
| is_duplicate | TINYINT(1) | 是否为重复文本（1=是）|
| created_at | DATETIME | 记录创建时间 |

### sentiment_results 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 主键（自增）|
| processed_text_id | BIGINT | 关联的 processed_texts.id |
| sentiment_label | VARCHAR(20) | 情感标签（positive/negative/neutral/mixed）|
| confidence | FLOAT | 置信度（0.0 ~ 1.0）|
| reasoning | TEXT | 模型给出的分析理由 |
| model_name | VARCHAR(100) | 使用的模型名称 |
| created_at | DATETIME | 记录创建时间 |

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

## 查询示例

```sql
-- 查看各情感标签分布
SELECT sentiment_label, COUNT(*) AS count
FROM sentiment_results
GROUP BY sentiment_label
ORDER BY count DESC;

-- 查看高置信度的负面情感文本
SELECT pt.source_type, pt.source_id, pt.processed_text,
       sr.sentiment_label, sr.confidence, sr.reasoning
FROM processed_texts pt
JOIN sentiment_results sr ON sr.processed_text_id = pt.id
WHERE sr.sentiment_label = 'negative'
  AND sr.confidence >= 0.9
ORDER BY sr.confidence DESC
LIMIT 20;

-- 按来源类型统计情感分布
SELECT pt.source_type, sr.sentiment_label, COUNT(*) AS count
FROM processed_texts pt
JOIN sentiment_results sr ON sr.processed_text_id = pt.id
GROUP BY pt.source_type, sr.sentiment_label
ORDER BY pt.source_type, count DESC;

-- 查看预处理统计
SELECT
    source_type,
    COUNT(*) AS total,
    SUM(is_duplicate) AS duplicates,
    COUNT(DISTINCT language) AS languages
FROM processed_texts
GROUP BY source_type;
```

## 注意事项

1. **运行顺序**：请先运行 Step-1-Code 采集数据，再运行本项目处理数据。
2. **API 费用**：DeepSeek API 按调用量计费，建议先用少量数据测试。
3. **速率限制**：`request_delay` 参数控制请求间隔，避免触发 API 速率限制。
4. **断点续跑**：情感分析模块只处理尚未分析的文本（`sentiment_results` 中无记录的），可随时中断后继续运行。
5. **语言检测准确性**：短文本的语言检测可能不准确，可通过 `min_text_length` 参数过滤过短文本。

## 许可证

本项目使用 MIT 许可证。