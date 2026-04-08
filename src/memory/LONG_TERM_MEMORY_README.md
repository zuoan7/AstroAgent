# 长期记忆功能说明

## 概述

长期记忆模块为天文Agent添加了跨会话的用户画像存储能力，能够从对话中自动提取用户偏好、习惯和约束，并在后续对话中应用这些信息，提供更加个性化的服务。

## 核心特性

### 1. 用户画像存储
- **存储方式**: SQLite数据库 (`./long_term_memory/user_profiles.sqlite`)
- **用户ID**: 支持多用户，默认使用 `anonymous`
- **数据维度**:
  - **偏好**: 回答风格、知识水平等
  - **习惯**: 常问话题、关注的天体等
  - **约束**: 内容限制、长度限制等

### 2. 自动信息提取
从每次对话中自动提取:
- 显式表述（如"我喜欢简短回答"）
- 隐式偏好（如关注特定天体）
- 约束条件（如"不要用专业术语"）

### 3. Prompt注入
用户画像自动注入到Agent的Prompt中，影响回答风格和内容选择。

## 数据结构

```python
@dataclass
class UserProfile:
    user_id: str              # 用户ID
    preferences: Dict[str, Any]  # 偏好设置
    habits: Dict[str, Any]      # 行为习惯
    constraints: List[str]      # 约束条件
    created_at: str           # 创建时间
    updated_at: str           # 更新时间
```

## 使用方式

### 方式1: 通过API

#### 获取用户画像
```bash
curl http://localhost:8000/profile
```

#### 删除用户画像
```bash
curl -X DELETE http://localhost:8000/profile
```

#### 指定用户ID查询
```bash
curl http://localhost:8000/profile?user_id=user_123
```

### 方式2: 编程接口

```python
from agent import AstroAgent

# 创建Agent（自动初始化长期记忆）
agent = AstroAgent(user_id="my_user_id")

# 查询时自动提取并保存用户画像
response = agent.generate_events("我喜欢详细的回答，讲讲黑洞")

# 手动访问用户画像
profile = agent.long_term_memory.load_profile("my_user_id")
print(profile.preferences)
```

### 方式3: 直接使用LongTermMemory类

```python
from memory import LongTermMemory

ltm = LongTermMemory("./long_term_memory/my_db.sqlite")

# 提取信息
extracted = ltm.extract_from_conversation(
    user_message="我喜欢简短回答",
    assistant_message="好的，我会简洁回答"
)

# 合并更新
ltm.merge_and_update("user_123", extracted)

# 加载画像
profile = ltm.load_profile("user_123")

# 格式化用于Prompt
formatted = ltm.format_profile_for_prompt("user_123")
```

## 工作流程

```
用户查询
  ↓
加载长期记忆（用户画像）
  ↓
格式化用户画像
  ↓
注入到Prompt（{user_profile}）
  ↓
Agent推理 + 工具调用
  ↓
生成回答
  ↓
提取新偏好信息
  ↓
更新长期记忆
```

## 配置参数

在 `config.py` 中配置:

```python
# 长期记忆配置
LONG_TERM_MEMORY_PATH: str = "./long_term_memory/user_profiles.sqlite"
DEFAULT_USER_ID: str = "anonymous"
```

## 信息提取规则

### 偏好提取
- `简短/简单` → `response_style: 简短`
- `详细/深入` → `response_style: 详细`
- `专业` → `knowledge_level: 专业`
- `通俗/易懂` → `knowledge_level: 通俗`

### 约束提取
- `不要...术语` → `constraints: ["避免使用专业术语"]`
- `不要超过...字` → `constraints: ["控制回答长度"]`

### 习惯提取
- 检测天体关键词: 火星、木星、黑洞、星系、星云等
- 累计到 `frequent_topics` 列表

## 示例对话

### 场景1: 表达偏好
```
用户: 我喜欢简短一点的回答，不要太啰嗦
助手: 好的，我会简洁回答
→ 保存: preferences["response_style"] = "简短"
```

### 场景2: 表达兴趣
```
用户: 我对黑洞和星系很感兴趣
助手: 好的，我来讲讲黑洞...
→ 保存: habits["frequent_topics"] = ["黑洞", "星系"]
```

### 场景3: 表达约束
```
用户: 不要用太多专业术语，我是初学者
助手: 明白，我会用通俗易懂的语言
→ 保存: constraints.append("避免使用专业术语")
```

## 数据管理

### 查看数据库内容
```bash
sqlite3 long_term_memory/user_profiles.sqlite
SELECT * FROM user_profiles;
```

### 清空所有画像
```bash
rm long_term_memory/user_profiles.sqlite
```

### 备份画像数据
```bash
cp long_term_memory/user_profiles.sqlite long_term_memory/backup.sqlite
```

## 测试

运行测试脚本:
```bash
python test_long_term_memory.py
```

运行演示脚本:
```bash
python demo_long_term_memory.py
```

## 扩展建议

### 1. 增强提取能力
- 使用LLM进行语义提取（而非规则）
- 添加情感分析（用户情绪偏好）
- 追踪对话模式（提问时间、频率）

### 2. 智能更新
- 置信度评分（低置信度不存储）
- 时间衰减（长期未使用的偏好降低权重）
- 冲突解决（矛盾的偏好处理）

### 3. 向量化检索
- 复用RAG的向量库
- 根据当前问题检索相关偏好
- 支持模糊匹配和语义联想

### 4. 隐私保护
- 数据加密存储
- 用户可选择性删除画像
- 定期清理过期数据

## 注意事项

1. **SQLite是内置库，无需安装额外依赖**
2. 长期记忆在Agent实例化时自动初始化
3. 每次对话完成后自动提取并更新
4. 用户画像跨会话持久化保存
5. 可通过API手动查看和删除画像

## 故障排查

### 数据库初始化失败
- 检查 `./long_term_memory/` 目录权限
- 确保SQLite可写权限

### 画像未自动更新
- 检查日志是否有提取错误
- 验证对话内容是否匹配提取规则

### Prompt注入不生效
- 检查 `prompt_template.txt` 是否包含 `{user_profile}`
- 确认StreamingService已集成长期记忆
