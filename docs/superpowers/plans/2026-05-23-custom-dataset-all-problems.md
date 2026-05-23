# Custom Dataset Support for All Coordinate Problems — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend custom dataset upload from TSP-only to all 9 coordinate-based problem types (CVRP, SDVRP, mTSP, OP, PDP, PCTSP, SPCTSP, VRPTW, plus refactoring TSP).

**Architecture:** The upload endpoint gains a `problem_type` parameter; `parse_dataset` becomes problem-aware; `BaseTrainer` gains a shared `load_custom_dataset()` that all trainers call; each trainer adds `_inject_custom_data()` to overwrite TensorDict keys after `env.reset()`. The frontend dynamically shows/hides fields and disables upload for ATSP/FFSP.

**Tech Stack:** Flask, PyTorch, TensorDict (rl4co), vanilla JS in index.html, pytest.

---

## File Map

| File | Change |
|---|---|
| `app_files.py` | Extend `parse_dataset` + `upload_dataset` + `list_datasets` |
| `tests/test_parse_dataset.py` | Extend existing test file |
| `modules/rl_training/base_trainer.py` | Add `load_custom_dataset()`, init `custom_dataset_data = None` |
| `modules/rl_training/tsp_trainer.py` | Remove own `load_custom_dataset`, add `custom_dataset` property |
| `modules/rl_training/cvrp_trainer.py` | Add `load_custom_dataset` call + `_inject_custom_data` |
| `modules/rl_training/sdvrp_trainer.py` | Same pattern as CVRP |
| `modules/rl_training/mtsp_trainer.py` | Add `load_custom_dataset` call + `_inject_custom_data` |
| `modules/rl_training/op_trainer.py` | Add `load_custom_dataset` call + `_inject_custom_data` |
| `modules/rl_training/pdp_trainer.py` | Add `load_custom_dataset` call + `_inject_custom_data` |
| `modules/rl_training/pctsp_trainer.py` | Add `load_custom_dataset` call + `_inject_custom_data` |
| `modules/rl_training/spctsp_trainer.py` | Add `load_custom_dataset` call + `_inject_custom_data` |
| `modules/rl_training/vrptw_trainer.py` | Add `load_custom_dataset` call + `_inject_custom_data` |
| `templates/index.html` | Dynamic description, ATSP/FFSP disable, pass `problem_type` |

---

## Task 1: Extend `parse_dataset` and `upload_dataset` in `app_files.py`

**Files:**
- Modify: `app_files.py`
- Modify: `tests/test_parse_dataset.py`

### Background

Current `parse_dataset(content, file_ext)` returns a plain list of coordinates. It needs to:
1. Accept `problem_type` parameter
2. Return a dict with `coordinates` + optional fields, instead of a plain list
3. Validate problem-specific constraints

Current `upload_dataset` stores `num_cities`; change to `num_loc`. It also needs to receive `problem_type` from the form and write it to the stored JSON.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_parse_dataset.py`:

```python
class TestParseDatasetMultiProblem:
    """测试多问题类型的数据集解析"""

    def test_tsp_returns_dict(self):
        """TSP 解析结果是 dict"""
        content = '{"coordinates": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}'
        result = parse_dataset(content, 'json', 'tsp')
        assert isinstance(result, dict)
        assert result['coordinates'] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    def test_cvrp_with_demands(self):
        """CVRP 解析含 demands 字段"""
        content = '{"coordinates": [[0.1, 0.2], [0.3, 0.4]], "demands": [0.3, 0.5]}'
        result = parse_dataset(content, 'json', 'cvrp')
        assert result['demands'] == [0.3, 0.5]
        assert result['coordinates'] == [[0.1, 0.2], [0.3, 0.4]]

    def test_cvrp_demands_out_of_range(self):
        """CVRP demands 超出 (0,1] 范围时返回 None"""
        content = '{"coordinates": [[0.1, 0.2], [0.3, 0.4]], "demands": [1.5, 0.5]}'
        result = parse_dataset(content, 'json', 'cvrp')
        assert result is None

    def test_pdp_odd_coordinates_fails(self):
        """PDP 奇数个坐标返回 None"""
        content = '{"coordinates": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}'
        result = parse_dataset(content, 'json', 'pdp')
        assert result is None

    def test_pdp_even_coordinates_ok(self):
        """PDP 偶数个坐标通过"""
        content = '{"coordinates": [[0.1, 0.2], [0.3, 0.4]]}'
        result = parse_dataset(content, 'json', 'pdp')
        assert result is not None
        assert len(result['coordinates']) == 2

    def test_vrptw_with_time_windows(self):
        """VRPTW 解析含 time_windows"""
        content = '{"coordinates": [[0.1, 0.2]], "time_windows": [[0.0, 0.5]]}'
        result = parse_dataset(content, 'json', 'vrptw')
        assert result['time_windows'] == [[0.0, 0.5]]

    def test_vrptw_invalid_time_window(self):
        """VRPTW time_window start >= end 时返回 None"""
        content = '{"coordinates": [[0.1, 0.2]], "time_windows": [[0.8, 0.3]]}'
        result = parse_dataset(content, 'json', 'vrptw')
        assert result is None

    def test_op_with_prizes(self):
        """OP 解析含 prizes"""
        content = '{"coordinates": [[0.1, 0.2], [0.3, 0.4]], "prizes": [1.0, 2.0]}'
        result = parse_dataset(content, 'json', 'op')
        assert result['prizes'] == [1.0, 2.0]

    def test_op_negative_prize_fails(self):
        """OP prizes 包含负值时返回 None"""
        content = '{"coordinates": [[0.1, 0.2]], "prizes": [-0.1]}'
        result = parse_dataset(content, 'json', 'op')
        assert result is None

    def test_txt_still_works_for_tsp(self):
        """TXT 格式仍可用于 TSP"""
        result = parse_dataset("0.1 0.2\n0.3 0.4", 'txt', 'tsp')
        assert result is not None
        assert result['coordinates'] == [[0.1, 0.2], [0.3, 0.4]]

    def test_depot_optional_field_parsed(self):
        """depot 字段可选，解析后保留"""
        content = '{"coordinates": [[0.1, 0.2]], "depot": [0.5, 0.5]}'
        result = parse_dataset(content, 'json', 'cvrp')
        assert result['depot'] == [0.5, 0.5]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_parse_dataset.py::TestParseDatasetMultiProblem -v
```

Expected: 11 FAILs (TypeError: parse_dataset takes 2 positional arguments...)

- [ ] **Step 3: Rewrite `parse_dataset` in `app_files.py`**

Replace the existing `parse_dataset` function (lines 26–84) with:

```python
def parse_dataset(content, file_ext, problem_type='tsp'):
    """解析数据集文件，返回包含 coordinates 等字段的 dict，失败返回 None。"""
    try:
        # --- 解析原始坐标 ---
        if file_ext == 'json':
            data = json.loads(content)
            if 'coordinates' not in data:
                return None
            coordinates = data['coordinates']
        elif file_ext == 'txt':
            coordinates = []
            for line in content.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            coordinates.append([float(parts[0]), float(parts[1])])
                        except ValueError:
                            continue
            if not coordinates:
                return None
            data = {}
        elif file_ext == 'tsp':
            coordinates = []
            in_section = False
            for line in content.strip().split('\n'):
                line = line.strip()
                if line.startswith('NODE_COORD_SECTION'):
                    in_section = True
                    continue
                if line in ['EOF', 'DISPLAY_DATA_SECTION', 'EDGE_WEIGHT_SECTION']:
                    break
                if in_section and line:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            coordinates.append([float(parts[1]), float(parts[2])])
                        except (ValueError, IndexError):
                            continue
            if not coordinates:
                return None
            data = {}
        else:
            return None

        if not coordinates:
            return None

        # --- 问题类型专用校验 ---
        if problem_type == 'pdp':
            if len(coordinates) % 2 != 0:
                return None  # PDP 坐标数必须是偶数

        # --- 构建结果 dict ---
        result = {'coordinates': coordinates}

        # depot（可选，所有问题通用）
        if isinstance(data, dict) and data.get('depot'):
            result['depot'] = data['depot']

        # demands（CVRP / SDVRP / VRPTW）
        if problem_type in ('cvrp', 'sdvrp', 'vrptw') and isinstance(data, dict) and data.get('demands'):
            demands = data['demands']
            if any(d <= 0 or d > 1 for d in demands):
                return None  # demands 必须在 (0, 1]
            result['demands'] = demands

        # prizes（OP / PCTSP / SPCTSP）
        if problem_type in ('op', 'pctsp', 'spctsp') and isinstance(data, dict) and data.get('prizes'):
            prizes = data['prizes']
            if any(p < 0 for p in prizes):
                return None  # prizes 不能为负
            result['prizes'] = prizes

        # penalties（PCTSP / SPCTSP）
        if problem_type in ('pctsp', 'spctsp') and isinstance(data, dict) and data.get('penalties'):
            result['penalties'] = data['penalties']

        # time_windows（VRPTW）
        if problem_type == 'vrptw' and isinstance(data, dict) and data.get('time_windows'):
            tws = data['time_windows']
            for tw in tws:
                if len(tw) != 2 or tw[0] >= tw[1]:
                    return None
            result['time_windows'] = tws

        # service_times（VRPTW）
        if problem_type == 'vrptw' and isinstance(data, dict) and data.get('service_times'):
            result['service_times'] = data['service_times']

        return result

    except Exception as e:
        logger.error(f"解析数据集错误: {str(e)}", exc_info=True)
        return None
```

- [ ] **Step 4: Fix existing tests that call `parse_dataset` with 2 args**

In `tests/test_parse_dataset.py`, update all calls in `TestParseDataset` to pass `problem_type='tsp'`:

```python
# 原来: parse_dataset(json_content, 'json')
# 改为: parse_dataset(json_content, 'json', 'tsp')
```

Also fix return type assertions — existing tests check `result[0] == [1.0, 2.0]`, but now `result` is a dict. Change them to:
```python
assert result['coordinates'][0] == [1.0, 2.0]
assert result['coordinates'][2] == [5.0, 6.0]
assert len(result['coordinates']) == 3
# etc.
```

Full updated `TestParseDataset`:

```python
class TestParseDataset:
    def test_parse_json_format(self):
        content = '{"coordinates": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}'
        result = parse_dataset(content, 'json', 'tsp')
        assert result is not None
        assert len(result['coordinates']) == 3
        assert result['coordinates'][0] == [1.0, 2.0]
        assert result['coordinates'][2] == [5.0, 6.0]

    def test_parse_txt_format(self):
        result = parse_dataset("1.0 2.0\n3.0 4.0\n5.0 6.0", 'txt', 'tsp')
        assert result is not None
        assert len(result['coordinates']) == 3
        assert result['coordinates'][0] == [1.0, 2.0]

    def test_parse_txt_with_comments(self):
        txt_content = "# comment\n1.0 2.0\n# another\n3.0 4.0"
        result = parse_dataset(txt_content, 'txt', 'tsp')
        assert result is not None
        assert len(result['coordinates']) == 2

    def test_parse_tsp_format(self):
        tsp_content = "NAME : test\nNODE_COORD_SECTION\n1 10.0 20.0\n2 30.0 40.0\n3 50.0 60.0\nEOF"
        result = parse_dataset(tsp_content, 'tsp', 'tsp')
        assert result is not None
        assert len(result['coordinates']) == 3
        assert result['coordinates'][0] == [10.0, 20.0]
        assert result['coordinates'][2] == [50.0, 60.0]

    def test_parse_invalid_json(self):
        result = parse_dataset('{"invalid": json}', 'json', 'tsp')
        assert result is None

    def test_parse_empty_content(self):
        result = parse_dataset('', 'txt', 'tsp')
        assert result is None

    def test_parse_json_missing_coordinates(self):
        result = parse_dataset('{"data": [[1.0, 2.0]]}', 'json', 'tsp')
        assert result is None
```

- [ ] **Step 5: Update `upload_dataset` endpoint in `app_files.py`**

In the `upload_dataset` function, make the following changes:

1. Read `problem_type` from form data (after the file checks):
```python
problem_type = request.form.get('problem_type', 'tsp').lower()
```

2. Pass it to `parse_dataset`:
```python
parsed = parse_dataset(file_content, file_ext, problem_type)
if parsed is None or not parsed.get('coordinates'):
    return jsonify({'success': False, 'message': '解析数据集失败，请检查文件格式'}), 400
coordinates = parsed['coordinates']
```

3. Store the full parsed result in the JSON file (replace the `json.dump` block):
```python
dataset_record = {
    'dataset_id': dataset_id,
    'user_id': user_id,
    'problem_type': problem_type,
    'filename': filename,
    'coordinates': coordinates,
    'num_loc': len(coordinates),
    'upload_time': datetime.now().isoformat(),
}
# 存储可选字段
for field in ('depot', 'demands', 'prizes', 'penalties', 'time_windows', 'service_times'):
    if field in parsed:
        dataset_record[field] = parsed[field]

with open(dataset_path, 'w') as f:
    json.dump(dataset_record, f)
```

4. Update the success response message and `dataset_info`:
```python
return jsonify({
    'success': True,
    'message': f'数据集上传成功！包含 {len(coordinates)} 个节点',
    'dataset_id': dataset_id,
    'dataset_info': {
        'filename': filename,
        'problem_type': problem_type,
        'num_loc': len(coordinates),
        'coord_range': coord_range,
    }
})
```

- [ ] **Step 6: Update `list_datasets` to return `problem_type` and `num_loc`**

In `list_datasets`, update the dict appended to `datasets`:
```python
datasets.append({
    'dataset_id': dataset_data.get('dataset_id'),
    'filename': dataset_data.get('filename'),
    'problem_type': dataset_data.get('problem_type', 'tsp'),
    'num_loc': dataset_data.get('num_loc', dataset_data.get('num_cities', 0)),
    'upload_time': dataset_data.get('upload_time'),
})
```

(The `dataset_data.get('num_cities', 0)` fallback handles old TSP records that used `num_cities`.)

- [ ] **Step 7: Run all tests**

```bash
pytest tests/test_parse_dataset.py -v
```

Expected: all tests PASS (7 original + 11 new = 18 total).

- [ ] **Step 8: Commit**

```bash
git add app_files.py tests/test_parse_dataset.py
git commit -m "feat: extend parse_dataset and upload_dataset to support all coordinate problem types"
```

---

## Task 2: Add `load_custom_dataset` to `BaseTrainer`, refactor `TSPTrainer`

**Files:**
- Modify: `modules/rl_training/base_trainer.py`
- Modify: `modules/rl_training/tsp_trainer.py`

### Background

`BaseTrainer.__init__` currently does not initialize `custom_dataset_data`. `TSPTrainer` has its own `load_custom_dataset` and a plain `self.custom_dataset` attribute. After this task: the method lives in `BaseTrainer`, `TSPTrainer` delegates to it via a `@property`.

- [ ] **Step 1: Add `custom_dataset_data = None` to `BaseTrainer.__init__`**

In `modules/rl_training/base_trainer.py`, find the `BaseTrainer.__init__` method. After the last `self.xxx = ...` attribute assignment (before any method calls), add:

```python
self.custom_dataset_data = None
```

- [ ] **Step 2: Add `load_custom_dataset` method to `BaseTrainer`**

Add the following method to the `BaseTrainer` class (place it before `initialize_environment`):

```python
def load_custom_dataset(self):
    """加载用户上传的自定义数据集（各 trainer 在 __init__ 末尾调用）。"""
    dataset_mode = self.config.get('dataset_mode', 'random')
    dataset_id = self.config.get('dataset_id', None)
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

    stored_type = data.get('problem_type', 'tsp')
    if stored_type != problem_type:
        self.send_message('info',
            f'⚠️ 数据集类型({stored_type})与当前问题({problem_type})不匹配，将使用随机生成')
        return

    self.custom_dataset_data = data
    self.num_loc = len(data['coordinates'])
    self.send_message('info',
        f'✅ 已加载自定义数据集: {data["filename"]} ({self.num_loc} 个节点)')
```

- [ ] **Step 3: Refactor `TSPTrainer`**

In `modules/rl_training/tsp_trainer.py`:

1. Remove the `load_custom_dataset` method (lines 35–55).
2. In `__init__`, replace:
   ```python
   self.custom_dataset = None
   self.load_custom_dataset()
   ```
   with:
   ```python
   self.load_custom_dataset()
   ```
3. Add a `custom_dataset` property after `__init__`:
   ```python
   @property
   def custom_dataset(self):
       if self.custom_dataset_data:
           return self.custom_dataset_data['coordinates']
       return None
   ```

The rest of `TSPTrainer` (`initialize_environment`, `create_model`, `generate_visualizations`) stays unchanged — they already use `self.custom_dataset` which the property now provides.

- [ ] **Step 4: Smoke-test with pytest**

```bash
pytest tests/test_parse_dataset.py -v
```

Expected: 18 PASS (no regressions).

- [ ] **Step 5: Commit**

```bash
git add modules/rl_training/base_trainer.py modules/rl_training/tsp_trainer.py
git commit -m "refactor: move load_custom_dataset to BaseTrainer, make TSPTrainer.custom_dataset a property"
```

---

## Task 3: CVRP and SDVRP — load + inject

**Files:**
- Modify: `modules/rl_training/cvrp_trainer.py`
- Modify: `modules/rl_training/sdvrp_trainer.py`

### Background

Both CVRP and SDVRP share the same TensorDict structure:
- `locs`: `[B, num_loc+1, 2]` — depot at index 0, then customers
- `demand`: `[B, num_loc]` — customer demands (depot excluded)

- [ ] **Step 1: Update `CVRPTrainer.__init__`**

In `cvrp_trainer.py`, at the end of `CVRPTrainer.__init__` (after `self.num_vehicles = ...`), add:

```python
self.load_custom_dataset()
```

- [ ] **Step 2: Add `_inject_custom_data` to `CVRPTrainer`**

Add the following method to `CVRPTrainer` (before `generate_visualizations`):

```python
def _inject_custom_data(self, td):
    """将自定义数据集注入 TensorDict。"""
    data = self.custom_dataset_data
    coords = torch.tensor(data['coordinates'], dtype=torch.float32)  # [N, 2]

    if data.get('depot'):
        depot = torch.tensor(data['depot'], dtype=torch.float32)  # [2]
    else:
        depot = td['locs'][0, 0].cpu()  # 从 env.reset 结果取 depot

    locs = torch.cat([depot.unsqueeze(0), coords], dim=0)  # [N+1, 2]
    td['locs'] = locs.unsqueeze(0).to(self.device)  # [1, N+1, 2]

    if data.get('demands'):
        demand = torch.tensor(data['demands'], dtype=torch.float32)  # [N]
        td['demand'] = demand.unsqueeze(0).to(self.device)  # [1, N]

    return td
```

- [ ] **Step 3: Update `CVRPTrainer.generate_visualizations`**

Replace the line:
```python
td_init = env.reset(batch_size=[3]).to(self.device)
```
with:
```python
if self.custom_dataset_data:
    td_init = env.reset(batch_size=[1]).to(self.device)
    td_init = self._inject_custom_data(td_init)
    self.send_message('info', f'✅ 在上传的CVRP数据集上进行测试（{self.num_loc}个客户）')
else:
    td_init = env.reset(batch_size=[3]).to(self.device)
```

- [ ] **Step 4: Update `SDVRPTrainer` identically**

In `sdvrp_trainer.py`:

1. At the end of `SDVRPTrainer.__init__` (after `self.allow_split = True`), add:
   ```python
   self.load_custom_dataset()
   ```

2. Add `_inject_custom_data` (identical to CVRP's):
   ```python
   def _inject_custom_data(self, td):
       data = self.custom_dataset_data
       coords = torch.tensor(data['coordinates'], dtype=torch.float32)

       if data.get('depot'):
           depot = torch.tensor(data['depot'], dtype=torch.float32)
       else:
           depot = td['locs'][0, 0].cpu()

       locs = torch.cat([depot.unsqueeze(0), coords], dim=0)
       td['locs'] = locs.unsqueeze(0).to(self.device)

       if data.get('demands'):
           demand = torch.tensor(data['demands'], dtype=torch.float32)
           td['demand'] = demand.unsqueeze(0).to(self.device)

       return td
   ```

3. In `SDVRPTrainer.generate_visualizations`, find `td_init = env.reset(batch_size=[3]).to(device)` and replace with:
   ```python
   if self.custom_dataset_data:
       td_init = env.reset(batch_size=[1]).to(device)
       td_init = self._inject_custom_data(td_init)
       self.send_message('info', f'✅ 在上传的SDVRP数据集上进行测试（{self.num_loc}个客户）')
   else:
       td_init = env.reset(batch_size=[3]).to(device)
   ```

- [ ] **Step 5: Commit**

```bash
git add modules/rl_training/cvrp_trainer.py modules/rl_training/sdvrp_trainer.py
git commit -m "feat: add custom dataset support to CVRPTrainer and SDVRPTrainer"
```

---

## Task 4: mTSP — load + inject

**Files:**
- Modify: `modules/rl_training/mtsp_trainer.py`

### Background

mTSP TensorDict: `locs` is `[B, num_loc+1, 2]` with depot at index 0. No extra optional fields.

- [ ] **Step 1: Update `MTSPTrainer.__init__`**

At the end of `MTSPTrainer.__init__` (after the `self.send_message` info line about config), add:

```python
self.load_custom_dataset()
```

- [ ] **Step 2: Add `_inject_custom_data` to `MTSPTrainer`**

```python
def _inject_custom_data(self, td):
    data = self.custom_dataset_data
    coords = torch.tensor(data['coordinates'], dtype=torch.float32)  # [N, 2]

    if data.get('depot'):
        depot = torch.tensor(data['depot'], dtype=torch.float32)
    else:
        depot = td['locs'][0, 0].cpu()

    locs = torch.cat([depot.unsqueeze(0), coords], dim=0)  # [N+1, 2]
    td['locs'] = locs.unsqueeze(0).to(self.device)

    return td
```

- [ ] **Step 3: Update `MTSPTrainer.generate_visualizations`**

In `generate_visualizations`, find:
```python
num_test_instances = min(3, self.batch_size)
td = env.reset(batch_size=[num_test_instances])
```

Replace with:
```python
if self.custom_dataset_data:
    td = env.reset(batch_size=[1]).to(self.device)
    td = self._inject_custom_data(td)
    self.send_message('info', f'✅ 在上传的mTSP数据集上进行测试（{self.num_loc}个城市）')
else:
    num_test_instances = min(3, self.batch_size)
    td = env.reset(batch_size=[num_test_instances])
```

- [ ] **Step 4: Commit**

```bash
git add modules/rl_training/mtsp_trainer.py
git commit -m "feat: add custom dataset support to MTSPTrainer"
```

---

## Task 5: OP — load + inject

**Files:**
- Modify: `modules/rl_training/op_trainer.py`

### Background

OP TensorDict: `locs` is `[B, num_loc+1, 2]` with depot at index 0. Optional: `prize` is `[B, num_loc+1]` where `prize[0]=0` for depot.

- [ ] **Step 1: Update `OPTrainer.__init__`**

At the end of `OPTrainer.__init__` (after the two `send_message` info lines), add:

```python
self.load_custom_dataset()
```

- [ ] **Step 2: Add `_inject_custom_data` to `OPTrainer`**

```python
def _inject_custom_data(self, td):
    data = self.custom_dataset_data
    coords = torch.tensor(data['coordinates'], dtype=torch.float32)  # [N, 2]

    if data.get('depot'):
        depot = torch.tensor(data['depot'], dtype=torch.float32)
    else:
        depot = td['locs'][0, 0].cpu()

    locs = torch.cat([depot.unsqueeze(0), coords], dim=0)  # [N+1, 2]
    td['locs'] = locs.unsqueeze(0).to(self.device)

    if data.get('prizes'):
        # prize[0]=0 for depot, then customer prizes
        prize = torch.tensor([0.0] + data['prizes'], dtype=torch.float32)  # [N+1]
        td['prize'] = prize.unsqueeze(0).to(self.device)

    return td
```

- [ ] **Step 3: Update `OPTrainer.generate_visualizations`**

Find:
```python
num_test_instances = min(3, self.batch_size)
td = env.reset(batch_size=[num_test_instances])
```

Replace with:
```python
if self.custom_dataset_data:
    td = env.reset(batch_size=[1]).to(self.device)
    td = self._inject_custom_data(td)
    self.send_message('info', f'✅ 在上传的OP数据集上进行测试（{self.num_loc}个地点）')
else:
    num_test_instances = min(3, self.batch_size)
    td = env.reset(batch_size=[num_test_instances])
```

- [ ] **Step 4: Commit**

```bash
git add modules/rl_training/op_trainer.py
git commit -m "feat: add custom dataset support to OPTrainer"
```

---

## Task 6: PDP — load + inject

**Files:**
- Modify: `modules/rl_training/pdp_trainer.py`

### Background

PDP TensorDict is different from other problems:
- `locs`: `[B, num_loc, 2]` — customer coords only (NO depot in locs)
- `depot`: `[B, 2]` — separate depot key

The `num_loc` must be even (pickup-delivery pairs). Custom dataset `coordinates` length must be even (validated in Task 1).

- [ ] **Step 1: Update `PDPTrainer.__init__`**

At the end of `PDPTrainer.__init__` (after the `send_message` info line), add:

```python
self.load_custom_dataset()
```

Also, after `load_custom_dataset()` call, if custom data is loaded, update `pdp_num_loc`:
```python
if self.custom_dataset_data:
    self.pdp_num_loc = self.num_loc
    self.num_pairs = self.pdp_num_loc // 2
```

- [ ] **Step 2: Add `_inject_custom_data` to `PDPTrainer`**

```python
def _inject_custom_data(self, td):
    data = self.custom_dataset_data
    coords = torch.tensor(data['coordinates'], dtype=torch.float32)  # [N, 2]

    if data.get('depot'):
        depot = torch.tensor(data['depot'], dtype=torch.float32)  # [2]
    else:
        depot = td['depot'][0].cpu()  # PDP keeps depot in separate key

    td['locs'] = coords.unsqueeze(0).to(self.device)    # [1, N, 2]
    td['depot'] = depot.unsqueeze(0).to(self.device)    # [1, 2]

    return td
```

- [ ] **Step 3: Update `PDPTrainer.generate_visualizations`**

Find:
```python
num_test_instances = min(3, self.batch_size)
td = env.reset(batch_size=[num_test_instances])
```

Replace with:
```python
if self.custom_dataset_data:
    td = env.reset(batch_size=[1]).to(self.device)
    td = self._inject_custom_data(td)
    self.send_message('info', f'✅ 在上传的PDP数据集上进行测试（{self.num_loc}个节点，{self.num_pairs}对取送货）')
else:
    num_test_instances = min(3, self.batch_size)
    td = env.reset(batch_size=[num_test_instances])
```

- [ ] **Step 4: Commit**

```bash
git add modules/rl_training/pdp_trainer.py
git commit -m "feat: add custom dataset support to PDPTrainer"
```

---

## Task 7: PCTSP and SPCTSP — load + inject

**Files:**
- Modify: `modules/rl_training/pctsp_trainer.py`
- Modify: `modules/rl_training/spctsp_trainer.py`

### Background

PCTSP TensorDict: `locs [B, num_loc+1, 2]`, `real_prize [B, num_loc+1]` (depot=0), `penalty [B, num_loc+1]` (depot=0).

SPCTSP additionally has `deterministic_prize [B, num_loc]` (no depot row).

- [ ] **Step 1: Update `PCTSPTrainer.__init__`**

At the end of `PCTSPTrainer.__init__` (after the two `send_message` lines), add:

```python
self.load_custom_dataset()
```

- [ ] **Step 2: Add `_inject_custom_data` to `PCTSPTrainer`**

```python
def _inject_custom_data(self, td):
    data = self.custom_dataset_data
    coords = torch.tensor(data['coordinates'], dtype=torch.float32)  # [N, 2]

    if data.get('depot'):
        depot = torch.tensor(data['depot'], dtype=torch.float32)
    else:
        depot = td['locs'][0, 0].cpu()

    locs = torch.cat([depot.unsqueeze(0), coords], dim=0)  # [N+1, 2]
    td['locs'] = locs.unsqueeze(0).to(self.device)

    if data.get('prizes'):
        real_prize = torch.tensor([0.0] + data['prizes'], dtype=torch.float32)  # [N+1]
        td['real_prize'] = real_prize.unsqueeze(0).to(self.device)

    if data.get('penalties'):
        penalty = torch.tensor([0.0] + data['penalties'], dtype=torch.float32)  # [N+1]
        td['penalty'] = penalty.unsqueeze(0).to(self.device)

    return td
```

- [ ] **Step 3: Update `PCTSPTrainer.generate_visualizations`**

Find:
```python
num_test = min(3, self.batch_size)
td = env.reset(batch_size=[num_test])
```

Replace with:
```python
if self.custom_dataset_data:
    td = env.reset(batch_size=[1]).to(self.device)
    td = self._inject_custom_data(td)
    self.send_message('info', f'✅ 在上传的PCTSP数据集上进行测试（{self.num_loc}个客户节点）')
else:
    num_test = min(3, self.batch_size)
    td = env.reset(batch_size=[num_test])
```

- [ ] **Step 4: Update `SPCTSPTrainer.__init__`**

At the end of `SPCTSPTrainer.__init__` (after `send_message` about deterministic/real prize), add:

```python
self.load_custom_dataset()
```

- [ ] **Step 5: Add `_inject_custom_data` to `SPCTSPTrainer`**

```python
def _inject_custom_data(self, td):
    data = self.custom_dataset_data
    coords = torch.tensor(data['coordinates'], dtype=torch.float32)  # [N, 2]

    if data.get('depot'):
        depot = torch.tensor(data['depot'], dtype=torch.float32)
    else:
        depot = td['locs'][0, 0].cpu()

    locs = torch.cat([depot.unsqueeze(0), coords], dim=0)  # [N+1, 2]
    td['locs'] = locs.unsqueeze(0).to(self.device)

    if data.get('prizes'):
        prizes = data['prizes']
        # deterministic_prize: [N] (no depot)
        det_prize = torch.tensor(prizes, dtype=torch.float32)
        td['deterministic_prize'] = det_prize.unsqueeze(0).to(self.device)
        # real_prize: [0.0 (depot)] + prizes → [N+1]
        real_prize = torch.tensor([0.0] + prizes, dtype=torch.float32)
        td['real_prize'] = real_prize.unsqueeze(0).to(self.device)

    if data.get('penalties'):
        penalty = torch.tensor([0.0] + data['penalties'], dtype=torch.float32)
        td['penalty'] = penalty.unsqueeze(0).to(self.device)

    return td
```

- [ ] **Step 6: Update `SPCTSPTrainer.generate_visualizations`**

Find:
```python
num_test = min(3, self.batch_size)
td = env.reset(batch_size=[num_test])
```

Replace with:
```python
if self.custom_dataset_data:
    td = env.reset(batch_size=[1]).to(self.device)
    td = self._inject_custom_data(td)
    self.send_message('info', f'✅ 在上传的SPCTSP数据集上进行测试（{self.num_loc}个客户节点）')
else:
    num_test = min(3, self.batch_size)
    td = env.reset(batch_size=[num_test])
```

- [ ] **Step 7: Commit**

```bash
git add modules/rl_training/pctsp_trainer.py modules/rl_training/spctsp_trainer.py
git commit -m "feat: add custom dataset support to PCTSPTrainer and SPCTSPTrainer"
```

---

## Task 8: VRPTW — load + inject

**Files:**
- Modify: `modules/rl_training/vrptw_trainer.py`

### Background

VRPTW TensorDict: `locs [B, num_loc+1, 2]`, `demand [B, num_loc+1]`, `time_windows [B, num_loc+1, 2]`, `service_time [B, num_loc+1]`. Index 0 is depot in all tensors.

For `time_windows`, if user provides customer windows, keep the depot's window from `env.reset()` (which knows the correct `max_time`).

- [ ] **Step 1: Update `VRPTWTrainer.__init__`**

At the end of `VRPTWTrainer.__init__` (after the validation checks and `send_message` warnings), add:

```python
self.load_custom_dataset()
```

- [ ] **Step 2: Add `_inject_custom_data` to `VRPTWTrainer`**

```python
def _inject_custom_data(self, td):
    data = self.custom_dataset_data
    coords = torch.tensor(data['coordinates'], dtype=torch.float32)  # [N, 2]

    if data.get('depot'):
        depot = torch.tensor(data['depot'], dtype=torch.float32)
    else:
        depot = td['locs'][0, 0].cpu()

    locs = torch.cat([depot.unsqueeze(0), coords], dim=0)  # [N+1, 2]
    td['locs'] = locs.unsqueeze(0).to(self.device)

    if data.get('demands'):
        # demand includes depot (=0)
        demand = torch.tensor([0.0] + data['demands'], dtype=torch.float32)  # [N+1]
        td['demand'] = demand.unsqueeze(0).to(self.device)

    if data.get('time_windows'):
        # keep depot's time window from env, replace customer rows
        depot_tw = td['time_windows'][0, 0:1].cpu()  # [1, 2]
        customer_tw = torch.tensor(data['time_windows'], dtype=torch.float32)  # [N, 2]
        full_tw = torch.cat([depot_tw, customer_tw], dim=0)  # [N+1, 2]
        td['time_windows'] = full_tw.unsqueeze(0).to(self.device)

    if data.get('service_times'):
        service_time = torch.tensor([0.0] + data['service_times'], dtype=torch.float32)  # [N+1]
        td['service_time'] = service_time.unsqueeze(0).to(self.device)

    return td
```

- [ ] **Step 3: Update `VRPTWTrainer.generate_visualizations`**

The current VRPTW `generate_visualizations` uses `env.reset(batch_size=[1])` (already batch=1). Find that line:

```python
td_init = env.reset(batch_size=[1]).to(self.device)
```

Replace with:
```python
td_init = env.reset(batch_size=[1]).to(self.device)
if self.custom_dataset_data:
    td_init = self._inject_custom_data(td_init)
    self.send_message('info', f'✅ 在上传的VRPTW数据集上进行测试（{self.num_loc}个客户）')
```

- [ ] **Step 4: Commit**

```bash
git add modules/rl_training/vrptw_trainer.py
git commit -m "feat: add custom dataset support to VRPTWTrainer"
```

---

## Task 9: Frontend — `templates/index.html`

**Files:**
- Modify: `templates/index.html`

### Changes

1. Section title: remove "仅TSP"
2. New JS function `updateDatasetSection(problemType)` — updates description text and disables controls for ATSP/FFSP
3. Hook into existing `problem-select` change listener
4. `uploadDataset()` — append `problem_type` to FormData
5. `showDatasetManager()` / dataset list rendering — add `problem_type` chip, sort matched first

- [ ] **Step 1: Update section title**

Find:
```html
<h3><span class="icon">📁</span> 第4步：数据集（可选，仅TSP）</h3>
<p style="color: #666; margin-bottom: 1rem; font-size: 0.95rem;">
    💡 默认使用随机生成数据集。如需使用自定义TSP数据集，请在此上传。
</p>
```

Replace with:
```html
<h3><span class="icon">📁</span> 第4步：数据集（可选）</h3>
<p id="dataset-description" style="color: #666; margin-bottom: 1rem; font-size: 0.95rem;">
    💡 默认使用随机生成数据集。如需使用自定义数据集，请在此上传。
</p>
```

- [ ] **Step 2: Add IDs to upload button and manager button**

Find the upload button:
```html
<button type="button" class="start-button" onclick="uploadDataset()" style="font-size: 1rem; padding: 0.8rem;">
    📤 上传数据集
</button>
```
Add `id="dataset-upload-btn"`:
```html
<button type="button" id="dataset-upload-btn" class="start-button" onclick="uploadDataset()" style="font-size: 1rem; padding: 0.8rem;">
    📤 上传数据集
</button>
```

Find the manager button:
```html
<button type="button" class="start-button" onclick="showDatasetManager()" style="margin-top: 1rem; font-size: 0.9rem; padding: 0.6rem; background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
    📂 管理已上传数据集
</button>
```
Add `id="dataset-manager-btn"`:
```html
<button type="button" id="dataset-manager-btn" class="start-button" onclick="showDatasetManager()" style="margin-top: 1rem; font-size: 0.9rem; padding: 0.6rem; background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
    📂 管理已上传数据集
</button>
```

- [ ] **Step 3: Add `updateDatasetSection` JS function**

Find the `<script>` block (near other JS functions). Add before the closing `</script>`:

```javascript
const DATASET_DESCRIPTIONS = {
    tsp:    '💡 支持格式：JSON（含 coordinates 字段）、TXT（每行 x y）、TSP（TSPLIB 格式）。',
    cvrp:   '💡 上传 JSON 文件。必填：coordinates（客户坐标列表）。可选：depot（仓库坐标）、demands（需求量，值域 0-1）。',
    sdvrp:  '💡 上传 JSON 文件。必填：coordinates（客户坐标列表）。可选：depot（仓库坐标）、demands（需求量，值域 0-1）。',
    mtsp:   '💡 上传 JSON 文件。必填：coordinates（城市坐标列表）。可选：depot（仓库坐标）。',
    op:     '💡 上传 JSON 文件。必填：coordinates（地点坐标列表）。可选：depot（仓库坐标）、prizes（各地点奖励值，≥0）。',
    pdp:    '💡 上传 JSON 文件。必填：coordinates（节点坐标，数量必须是偶数，前半为取货点，后半为送货点）。可选：depot（仓库坐标）。',
    pctsp:  '💡 上传 JSON 文件。必填：coordinates（客户坐标列表）。可选：depot、prizes（奖励）、penalties（惩罚）。',
    spctsp: '💡 上传 JSON 文件。必填：coordinates（客户坐标列表）。可选：depot、prizes（期望奖励）、penalties（惩罚）。',
    vrptw:  '💡 上传 JSON 文件。必填：coordinates（客户坐标列表）。可选：depot、demands（需求量，值域 0-1）、time_windows（每项 [start, end]）、service_times（服务时长）。',
    atsp:   '⛔ ATSP 使用代价矩阵而非坐标，暂不支持上传自定义数据集。',
    ffsp:   '⛔ FFSP 使用作业工时矩阵而非坐标，暂不支持上传自定义数据集。',
};

function updateDatasetSection(problemType) {
    const unsupported = ['atsp', 'ffsp'];
    const disabled = unsupported.includes(problemType);

    // 更新描述文字
    const desc = DATASET_DESCRIPTIONS[problemType] || DATASET_DESCRIPTIONS['tsp'];
    document.getElementById('dataset-description').textContent = desc;
    document.getElementById('dataset-description').style.color = disabled ? '#dc3545' : '#666';

    // 禁用/启用控件
    document.getElementById('dataset-mode').disabled = disabled;
    document.getElementById('dataset-file').disabled = disabled;
    document.getElementById('dataset-upload-btn').disabled = disabled;
    document.getElementById('dataset-manager-btn').disabled = disabled;

    // ATSP/FFSP 时强制切回 random 模式并隐藏上传区
    if (disabled) {
        document.getElementById('dataset-mode').value = 'random';
        document.getElementById('dataset-upload-section').style.display = 'none';
    }
}
```

- [ ] **Step 4: Hook `updateDatasetSection` into the problem-select change listener**

Find the existing `problem-select` change listener (around line 2714):
```javascript
document.getElementById('problem-select').addEventListener('change', (e) => {
```

Inside that listener, add a call to `updateDatasetSection`:
```javascript
document.getElementById('problem-select').addEventListener('change', (e) => {
    updateDatasetSection(e.target.value);
    // ... existing code below ...
```

Also call it once on page load (after the listener registration):
```javascript
updateDatasetSection(document.getElementById('problem-select').value);
```

- [ ] **Step 5: Pass `problem_type` in `uploadDataset()`**

Find the `uploadDataset` function and the FormData block:
```javascript
const formData = new FormData();
formData.append('dataset', file);
```

Add after the `formData.append('dataset', file)` line:
```javascript
formData.append('problem_type', document.getElementById('problem-select').value);
```

- [ ] **Step 6: Update dataset list rendering in `showDatasetManager`**

Find where each dataset card/row is rendered (around line 2585, the `selectDataset` button). The current render creates a button per dataset. Update to:

1. Add `problem_type` chip to each row
2. Sort: matched problem type first, unmatched shown but dimmed

Find the section that maps datasets to HTML (look for `dataset.num_cities` or `dataset.filename`). Replace the rendering logic with:

```javascript
const currentProblem = document.getElementById('problem-select').value;
// Sort: matched first
const sorted = [...datasets].sort((a, b) => {
    const aMatch = (a.problem_type || 'tsp') === currentProblem ? 0 : 1;
    const bMatch = (b.problem_type || 'tsp') === currentProblem ? 0 : 1;
    return aMatch - bMatch;
});

const listHtml = sorted.map(dataset => {
    const matched = (dataset.problem_type || 'tsp') === currentProblem;
    const numLoc = dataset.num_loc || dataset.num_cities || '?';
    const ptLabel = (dataset.problem_type || 'tsp').toUpperCase();
    return `
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:0.8rem; border-bottom:1px solid #eee;
                    opacity:${matched ? 1 : 0.45};">
            <div>
                <span style="display:inline-block; padding:2px 8px; border-radius:12px;
                             background:#667eea; color:#fff; font-size:0.75rem;
                             margin-right:0.5rem;">${ptLabel}</span>
                <strong>${dataset.filename}</strong>
                <span style="color:#999; font-size:0.85rem; margin-left:0.5rem;">${numLoc} 个节点</span>
                ${!matched ? '<span style="color:#dc3545; font-size:0.8rem; margin-left:0.5rem;">（与当前问题不匹配）</span>' : ''}
            </div>
            <div style="display:flex; gap:0.5rem;">
                <button onclick="selectDataset('${dataset.dataset_id}', '${dataset.filename}', ${numLoc})"
                        ${!matched ? 'disabled title="请切换到对应问题类型后再选择"' : ''}
                        style="padding:0.3rem 0.8rem; font-size:0.85rem; cursor:${matched ? 'pointer' : 'not-allowed'};">
                    选择
                </button>
                <button onclick="deleteDataset('${dataset.dataset_id}')"
                        style="padding:0.3rem 0.8rem; font-size:0.85rem; background:#dc3545; color:#fff;">
                    删除
                </button>
            </div>
        </div>`;
}).join('');
document.getElementById('datasets-list').innerHTML = listHtml || '<p style="text-align:center;color:#999;">暂无数据集</p>';
```

- [ ] **Step 7: Update the success info display after upload**

Find where `dataset.num_cities` is shown in `#dataset-size` (around line 1165):
```html
<p style="margin: 0.3rem 0; font-size: 0.9rem;"><strong>城市数量:</strong> <span id="dataset-size"></span></p>
```

Change label to "节点数量":
```html
<p style="margin: 0.3rem 0; font-size: 0.9rem;"><strong>节点数量:</strong> <span id="dataset-size"></span></p>
```

In `uploadDataset()`, update the display after successful upload — change `result.dataset_info.num_cities` to `result.dataset_info.num_loc`:
```javascript
document.getElementById('dataset-size').textContent = result.dataset_info.num_loc + ' 个节点';
```

- [ ] **Step 8: Run the app and manually verify**

```bash
python app.py
```

Check:
1. Switch problem to CVRP → description updates to CVRP format
2. Switch to ATSP → description shows "暂不支持", controls disabled
3. Upload a CVRP JSON file → success message shows "个节点"
4. Open dataset manager → shows `CVRP` chip, matched datasets come first

- [ ] **Step 9: Commit**

```bash
git add templates/index.html
git commit -m "feat: update dataset upload UI for all problem types, disable for ATSP/FFSP"
```

---

## Self-Review

**Spec coverage:**
- ✅ Part 1 (upload endpoint): Task 1
- ✅ Part 2 (BaseTrainer + TSPTrainer): Task 2
- ✅ Part 3 (each trainer injection): Tasks 3–8
- ✅ Part 4 (frontend): Task 9
- ✅ ATSP/FFSP disabled with message: Task 9 Step 3
- ✅ `problem_type` passed in upload: Task 9 Step 5
- ✅ Dataset list shows `problem_type`, matched first: Task 9 Step 6
- ✅ `num_cities` → `num_loc` renamed: Task 1 Step 5, Task 9 Step 7
- ✅ PDP even-coordinate validation: Task 1 (parse_dataset)
- ✅ PDP separate `depot` key: Task 6 Step 2

**Placeholder scan:** No TBD/TODO found.

**Type consistency:**
- `custom_dataset_data` initialized in `BaseTrainer.__init__` (Task 2 Step 1), used in all trainers (Tasks 3–8) ✅
- `_inject_custom_data(self, td)` signature consistent across all trainers ✅
- `load_custom_dataset()` defined in BaseTrainer (Task 2 Step 2), called in each trainer `__init__` (Tasks 3–8) ✅
- `dataset_info.num_loc` in upload response matches frontend read `result.dataset_info.num_loc` ✅
