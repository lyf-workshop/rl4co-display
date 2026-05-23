# 设计文档：为所有坐标类问题添加自定义数据集上传支持

**日期**：2026-05-23  
**状态**：待实现  
**涉及问题类型**：TSP（已有）、CVRP、SDVRP、mTSP、OP、PDP、PCTSP、SPCTSP、VRPTW  
**暂不支持**：ATSP、FFSP（数据结构特殊，后续单独处理）

---

## 背景

当前只有 TSP 支持上传自定义数据集。其余 9 种坐标类问题只能用随机生成数据训练。本次扩展将统一的上传功能推广到所有坐标类问题。

---

## 设计模式：混合模式

- `coordinates` 字段**必填**（节点坐标列表）
- 其余字段（`depot`、`demands`、`time_windows` 等）**可选**——有则注入 TensorDict，无则由环境随机生成
- 不做坐标归一化：训练和测试使用同一份坐标，尺度自洽

---

## 第一部分：上传格式与解析（`app_files.py`）

### 用户上传的 JSON 格式

```json
{
  "coordinates": [[x1, y1], [x2, y2], ...],
  "depot":        [xd, yd],
  "demands":      [d1, d2, ...],
  "time_windows": [[start1, end1], [start2, end2], ...],
  "prizes":       [p1, p2, ...],
  "penalties":    [pen1, pen2, ...],
  "service_times": [s1, s2, ...]
}
```

### 各问题类型的字段规则

| 问题类型 | 必填 | 可选 | 额外校验 |
|---|---|---|---|
| tsp | coordinates | — | — |
| cvrp / sdvrp | coordinates | depot, demands | demands 值在 (0, 1] |
| mtsp | coordinates | depot | — |
| op | coordinates | depot, prizes | prizes 值 ≥ 0 |
| pdp | coordinates | depot | coordinates 数量必须是偶数 |
| pctsp / spctsp | coordinates | depot, prizes, penalties | — |
| vrptw | coordinates | depot, demands, time_windows, service_times | time_windows 每项为 [start, end]，start < end |

### 端点变更

**`POST /api/upload_dataset`**
- 新增表单字段 `problem_type`（默认 `tsp`）
- `parse_dataset(content, file_ext, problem_type)` 新增 `problem_type` 参数
- 存储的 JSON 新增 `problem_type` 字段，`num_cities` 统一改为 `num_loc`
- 解析失败时返回具体错误原因（如"PDP 坐标数必须是偶数"）

**`GET /api/list_datasets`**
- 返回列表中每条记录新增 `problem_type` 字段

---

## 第二部分：BaseTrainer 通用加载（`base_trainer.py`）

在 `BaseTrainer` 中新增 `load_custom_dataset()` 方法，替换 `TSPTrainer` 中的同名方法：

```python
def load_custom_dataset(self):
    dataset_mode = self.config.get('dataset_mode', 'random')
    dataset_id   = self.config.get('dataset_id', None)
    problem_type = self.config.get('problem', 'tsp')

    self.custom_dataset_data = None

    if dataset_mode != 'upload' or not dataset_id:
        return

    path = os.path.join('datasets', f'user_{self.user_id}', f'{dataset_id}.json')
    if not os.path.exists(path):
        self.send_message('info', '⚠️ 数据集文件不存在，将使用随机生成')
        return

    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        self.send_message('info', f'⚠️ 读取数据集失败: {e}，将使用随机生成')
        return

    if data.get('problem_type') != problem_type:
        self.send_message('info',
            f'⚠️ 数据集类型({data["problem_type"]})与当前问题({problem_type})不匹配，将使用随机生成')
        return

    self.custom_dataset_data = data
    self.num_loc = len(data['coordinates'])
    self.send_message('info',
        f'✅ 已加载自定义数据集: {data["filename"]} ({self.num_loc} 个节点)')
```

`BaseTrainer.__init__` 中初始化 `self.custom_dataset_data = None`。  
各 trainer 在 `__init__` 末尾调用 `self.load_custom_dataset()`。

**TSPTrainer 改动**：
- 删除自己的 `load_custom_dataset()`
- `__init__` 中删除 `self.custom_dataset = None; self.load_custom_dataset()`，改为调用 `self.load_custom_dataset()`
- 增加属性 `self.custom_dataset`，作为 `self.custom_dataset_data['coordinates']` 的别名（保持 `generate_visualizations` 不变）：
  ```python
  @property
  def custom_dataset(self):
      if self.custom_dataset_data:
          return self.custom_dataset_data['coordinates']
      return None
  ```

---

## 第三部分：各 Trainer 的 TensorDict 注入

### 通用注入策略

在 `generate_visualizations()` 中：

```python
td_init = env.reset(batch_size=[1]).to(self.device)
if self.custom_dataset_data:
    td_init = self._inject_custom_data(td_init)
else:
    td_init = env.reset(batch_size=[3]).to(self.device)
```

有自定义数据时 batch_size=1（单实例），无自定义数据时 batch_size=3（默认行为）。

### depot 来源优先级

1. `custom_dataset_data['depot']` 若存在，使用之
2. 否则取 `td_init['locs'][0, 0]`（env.reset 随机生成的 depot）

### 各 Trainer 的 `_inject_custom_data` 注入字段

| Trainer | 始终注入 | 有则注入（字段名 → TD key） |
|---|---|---|
| TSP | `locs` | — |
| CVRP / SDVRP | `locs`（depot+coords 拼接） | `demands` → `demand` |
| mTSP | `locs`（depot+coords 拼接） | — |
| OP | `locs`（depot+coords 拼接） | `prizes` → `prize` |
| PDP | `locs`（depot+coords 拼接） | — |
| PCTSP | `locs`（depot+coords 拼接） | `prizes` → `real_prize`, `penalties` → `penalty` |
| SPCTSP | `locs`（depot+coords 拼接） | `prizes` → `deterministic_prize` + `real_prize`, `penalties` → `penalty` |
| VRPTW | `locs`（depot+coords 拼接） | `demands` → `demand`, `time_windows` → `time_windows`, `service_times` → `service_time` |

`locs` tensor shape：`[1, num_loc+1, 2]`（batch=1，depot 在 index 0）  
其余 tensor shape 与各 env 的 reset 输出一致（从 td_init 已有 tensor 的 shape 推断）

---

## 第四部分：前端改动（`templates/index.html`）

### 卡片标题
- "第4步：数据集（可选，仅TSP）" → "第4步：数据集（可选）"

### 动态描述
JS 函数 `updateDatasetDescription(problemType)` 在 `problem-select` change 事件时调用，更新说明文字：

| 问题 | 说明文字 |
|---|---|
| tsp | 支持格式：JSON（含 coordinates 字段）、TXT（每行 x y）、TSP（TSPLIB）|
| cvrp / sdvrp | 上传 JSON，必填 coordinates，可选 depot、demands（值域 0-1）|
| mtsp | 上传 JSON，必填 coordinates，可选 depot |
| op | 上传 JSON，必填 coordinates，可选 depot、prizes |
| pdp | 上传 JSON，必填 coordinates（必须是偶数个），可选 depot |
| pctsp / spctsp | 上传 JSON，必填 coordinates，可选 depot、prizes、penalties |
| vrptw | 上传 JSON，必填 coordinates，可选 depot、demands、time_windows、service_times |
| atsp / ffsp | 该问题类型暂不支持上传数据集 |

### 禁用逻辑（ATSP/FFSP）
```javascript
const unsupported = ['atsp', 'ffsp'];
const disabled = unsupported.includes(problemType);
document.getElementById('dataset-file').disabled = disabled;
document.getElementById('dataset-upload-btn').disabled = disabled;
document.getElementById('dataset-manager-btn').disabled = disabled;
document.getElementById('dataset-mode').disabled = disabled;
```

### 上传请求
`uploadDataset()` 的 FormData 新增：
```javascript
formData.append('problem_type', document.getElementById('problem-select').value);
```

### 数据集列表
`showDatasetManager()` 渲染每条记录时：
- 新增 `problem_type` 标签（chip 样式）
- 当前问题类型匹配的记录排在前面，不匹配的显示但置灰

---

## 改动文件清单

| 文件 | 改动类型 |
|---|---|
| `app_files.py` | 扩展解析器，新增 problem_type 支持 |
| `modules/rl_training/base_trainer.py` | 新增通用 `load_custom_dataset()` |
| `modules/rl_training/tsp_trainer.py` | 改用基类方法，`custom_dataset` 改为 property |
| `modules/rl_training/cvrp_trainer.py` | 加载 + 注入 |
| `modules/rl_training/sdvrp_trainer.py` | 加载 + 注入 |
| `modules/rl_training/mtsp_trainer.py` | 加载 + 注入 |
| `modules/rl_training/op_trainer.py` | 加载 + 注入 |
| `modules/rl_training/pdp_trainer.py` | 加载 + 注入 |
| `modules/rl_training/pctsp_trainer.py` | 加载 + 注入 |
| `modules/rl_training/spctsp_trainer.py` | 加载 + 注入 |
| `modules/rl_training/vrptw_trainer.py` | 加载 + 注入 |
| `templates/index.html` | 动态描述，禁用逻辑，传 problem_type，列表标签 |
