# Training Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复训练模块的四个问题：清理死代码、让子类 Trainer 使用算法注册表、让共享状态写入更安全、让 epoch 内绘图异步化。

**Architecture:** 所有 Trainer 子类的 `create_model()` 改为更新 `self.config` 后调用 `super().create_model()`，从而让算法注册表（REINFORCE/PPO/A2C）生效；`ProgressCallback` 的状态写入改为单次 `dict.update()` 减少竞争窗口，绘图改为后台线程避免阻塞 epoch 边界。

**Tech Stack:** Python 3.8+, Flask, RL4CO, PyTorch Lightning, threading

---

## 文件改动概览

| 文件 | 操作 |
|------|------|
| `modules/rl_training/training_functions.py` | 移至 `archive/`（死代码） |
| `modules/rl_training/tsp_trainer.py` | 修改 `create_model`：更新 config → 调用 super |
| `modules/rl_training/cvrp_trainer.py` | 删除 `create_model` override，消息移到 `initialize_environment` |
| `modules/rl_training/sdvrp_trainer.py` | 删除 `create_model` override，消息移到 `initialize_environment` |
| `modules/rl_training/base_trainer.py` | `ProgressCallback`：状态写入改 `dict.update()`；绘图改后台线程 |

> `ffsp_trainer.py` 的 `create_model` **保留**：它使用 MatNet（非 REINFORCE），有独立理由。

---

### Task 1: 归档死代码 `training_functions.py`

**Files:**
- Move: `modules/rl_training/training_functions.py` → `archive/training_functions_legacy.py`

**背景：** `training_functions.py` 里的 `real_rl4co_training()`、旧版 `ProgressCallback`、`create_route_animation()` 均已被 `modules/rl_training/__init__.py` 和 `base_trainer.py` 替代。`app.py` 通过 `from modules.rl_training import real_rl4co_training` 导入的是 `__init__.py` 里的版本，不是这个文件。

- [ ] **Step 1: 确认没有活跃 import**

```bash
grep -r "from modules.rl_training.training_functions" D:/Users/Administrator/Documents/GitHub/rl4co-display --include="*.py"
grep -r "import training_functions" D:/Users/Administrator/Documents/GitHub/rl4co-display --include="*.py"
```

期望输出：无结果（空）。如有结果，先处理这些 import 再继续。

- [ ] **Step 2: 移动文件**

```powershell
Move-Item `
  "D:\Users\Administrator\Documents\GitHub\rl4co-display\modules\rl_training\training_functions.py" `
  "D:\Users\Administrator\Documents\GitHub\rl4co-display\archive\training_functions_legacy.py"
```

- [ ] **Step 3: 验证应用启动不报错**

```powershell
cd D:\Users\Administrator\Documents\GitHub\rl4co-display
python -c "import app; print('OK')"
```

期望输出：含 `OK`，无 `ImportError`。

- [ ] **Step 4: Commit**

```bash
git add archive/training_functions_legacy.py modules/rl_training/training_functions.py
git commit -m "refactor: archive dead training_functions.py (replaced by base_trainer + __init__ dispatch)"
```

---

### Task 2: 修复 `TSPTrainer.create_model()` 使用算法注册表

**Files:**
- Modify: `modules/rl_training/tsp_trainer.py`（第 62-95 行）

**问题：** `TSPTrainer.create_model()` 硬编码 REINFORCE，忽略用户在 UI 选择的算法（PPO/A2C）。自定义数据集场景下还需调整数据量大小，所以不能直接删 override，而是改为更新 config 后调用 `super().create_model()`。

- [ ] **Step 1: 替换 `TSPTrainer.create_model()`**

打开 `modules/rl_training/tsp_trainer.py`，将第 62-95 行的整个 `create_model` 方法替换为：

```python
def create_model(self, env, policy):
    """创建适用于TSP的RL模型（委托给BaseTrainer使用算法注册表）"""
    if self.custom_dataset is not None:
        n = len(self.custom_dataset)
        # 根据数据集规模调整训练数据量，覆盖 config 供算法注册表读取
        self.config['batch_size'] = min(self.batch_size, n)
        self.config['train_data_size'] = min(10_000, n * 20)
        self.config['val_data_size'] = min(1_000, n * 2)
        self.send_message('info', f'使用自定义TSP数据集训练模式（{n}个城市）')
    else:
        self.send_message('info', f'使用随机生成TSP数据集训练模式（{self.num_loc}个城市）')
    return super().create_model(env, policy)
```

- [ ] **Step 2: 验证语法**

```powershell
python -c "from modules.rl_training.tsp_trainer import TSPTrainer; print('OK')"
```

期望输出：含 `OK`（或因 rl4co 未安装输出 warning，但无 SyntaxError）。

- [ ] **Step 3: Commit**

```bash
git add modules/rl_training/tsp_trainer.py
git commit -m "fix: TSPTrainer.create_model delegates to algorithm registry via super()"
```

---

### Task 3: 删除 `CVRPTrainer.create_model()` 的无效 override

**Files:**
- Modify: `modules/rl_training/cvrp_trainer.py`（第 42-58 行）

**问题：** `CVRPTrainer.create_model()` 只是用默认参数硬编码 REINFORCE，和父类行为完全相同，多余。删除后父类的算法注册表生效，PPO/A2C 可以被正确选中。

- [ ] **Step 1: 将训练模式消息移到 `initialize_environment`**

在 `CVRPTrainer.initialize_environment()` 末尾（`return env` 之前）加一行：

```python
self.send_message('info', f'CVRP训练模式（{self.num_loc}个客户，容量={self.vehicle_capacity}）')
```

完整方法变为：

```python
def initialize_environment(self):
    """初始化CVRP环境"""
    env = CVRPEnv(generator_params={
        'num_loc': self.num_loc,
        'vehicle_capacity': self.vehicle_capacity
    })
    self.send_message('info', f'CVRP训练模式（{self.num_loc}个客户，容量={self.vehicle_capacity}）')
    return env
```

- [ ] **Step 2: 删除整个 `create_model` 方法（第 42-58 行）**

删除以下代码块（含空行）：

```python
def create_model(self, env, policy):
    """创建适用于CVRP的RL模型"""
    from rl4co.models import REINFORCE
    
    model = REINFORCE(
        env,
        policy,
        baseline="rollout",
        batch_size=self.batch_size,
        train_data_size=10_000,
        val_data_size=1_000,
        optimizer_kwargs={"lr": self.learning_rate},
    )
    
    self.send_message('info', f'使用CVRP训练模式（{self.num_loc}个客户，容量={self.vehicle_capacity}）')
    
    return model
```

- [ ] **Step 3: 验证语法**

```powershell
python -c "from modules.rl_training.cvrp_trainer import CVRPTrainer; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add modules/rl_training/cvrp_trainer.py
git commit -m "fix: remove CVRPTrainer.create_model override, algorithm registry now used"
```

---

### Task 4: 删除 `SDVRPTrainer.create_model()` 的无效 override

**Files:**
- Modify: `modules/rl_training/sdvrp_trainer.py`（第 67-85 行）

**问题：** 同 Task 3 中的 CVRP，只是硬编码了 REINFORCE 加默认参数，多余。

- [ ] **Step 1: 将训练模式消息移到 `initialize_environment`**

找到 `SDVRPTrainer.initialize_environment()` 的 `return env` 行，在其之前插入：

```python
self.send_message('info',
    f'SDVRP训练模式（{self.num_loc}个客户，容量={self.vehicle_capacity}，'
    f'最大分割={self.max_split_per_customer}次）')
```

- [ ] **Step 2: 删除整个 `create_model` 方法（第 67-85 行）**

删除以下代码块：

```python
def create_model(self, env, policy):
    """创建适用于SDVRP的RL模型"""
    from rl4co.models import REINFORCE
    
    model = REINFORCE(
        env,
        policy,
        baseline="rollout",
        batch_size=self.batch_size,
        train_data_size=10_000,
        val_data_size=1_000,
        optimizer_kwargs={"lr": self.learning_rate},
    )
    
    self.send_message('info', 
        f'使用SDVRP训练模式（{self.num_loc}个客户，容量={self.vehicle_capacity}，'
        f'最大分割={self.max_split_per_customer}次）')
    
    return model
```

- [ ] **Step 3: 验证语法**

```powershell
python -c "from modules.rl_training.sdvrp_trainer import SDVRPTrainer; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add modules/rl_training/sdvrp_trainer.py
git commit -m "fix: remove SDVRPTrainer.create_model override, algorithm registry now used"
```

---

### Task 5: 修复 `ProgressCallback` 的状态写入竞争条件

**Files:**
- Modify: `modules/rl_training/base_trainer.py`（`ProgressCallback.on_train_epoch_end`，约第 278-285 行）

**问题：** 当前代码对 `training_status[session_id]` 连续赋值 5 个字段，读取线程（SSE endpoint）在这 5 次赋值之间可能读到部分更新的状态。改为单次 `dict.update()` 可以把竞争窗口从 5 次操作压缩到 1 次。

- [ ] **Step 1: 定位目标代码**

在 `base_trainer.py` 的 `on_train_epoch_end` 方法中，找到：

```python
        # 同步更新全局 training_status，供训练结束时 final_results 读取
        if self.training_status is not None and self.session_id in self.training_status:
            self.training_status[self.session_id]['loss'] = round(loss, 4)
            self.training_status[self.session_id]['reward'] = round(reward, 4)
            self.training_status[self.session_id]['best_reward'] = round(self.best_reward, 4)
            self.training_status[self.session_id]['epoch'] = epoch
            self.training_status[self.session_id]['progress'] = round(progress, 2)
```

- [ ] **Step 2: 替换为单次 update**

```python
        # 同步更新全局 training_status，供训练结束时 final_results 读取
        # 使用单次 dict.update() 减少并发读取时看到部分状态的窗口
        if self.training_status is not None and self.session_id in self.training_status:
            self.training_status[self.session_id].update({
                'loss': round(loss, 4),
                'reward': round(reward, 4),
                'best_reward': round(self.best_reward, 4),
                'epoch': epoch,
                'progress': round(progress, 2),
            })
```

- [ ] **Step 3: 验证语法**

```powershell
python -c "from modules.rl_training.base_trainer import ProgressCallback; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add modules/rl_training/base_trainer.py
git commit -m "fix: use dict.update() in ProgressCallback to reduce status write race window"
```

---

### Task 6: 让 epoch 内绘图异步，不阻塞训练

**Files:**
- Modify: `modules/rl_training/base_trainer.py`（`ProgressCallback.on_train_epoch_end` 中的绘图部分，约第 217-276 行）

**问题：** 每个 epoch 末尾都在 Lightning 训练线程上同步执行 `plt.savefig()`（磁盘 I/O + matplotlib 渲染），会延迟下一个 epoch 的启动。改为后台 daemon 线程执行，训练线程只负责准备数据，不等待 I/O 完成。

**注意：** `Agg` 后端下每个线程创建自己的 figure 是线程安全的；我们复制历史数据列表避免共享引用。

- [ ] **Step 1: 在 `base_trainer.py` 顶部确认已有 `import threading`**

检查文件顶部的 import 区域。如果没有 `import threading`，在 `import os` 附近加一行：

```python
import threading
```

- [ ] **Step 2: 在 `ProgressCallback` 类中新增 `_save_plot_async` 方法**

在 `on_train_epoch_end` 方法**之前**（即紧接在类方法区）插入以下私有方法：

```python
def _save_plot_async(self, epochs, losses, rewards, best_reward,
                     plot_path, plot_url, epoch):
    """在后台线程中渲染并保存训练曲线，避免阻塞 epoch 边界"""
    def _save():
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            ax1.plot(epochs, losses, 'b-o', linewidth=2, markersize=6, label='Loss')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('训练Loss变化曲线', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='upper right', fontsize=10)

            ax2.plot(epochs, rewards, 'g-o', linewidth=2, markersize=6, label='Reward')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Reward', fontsize=12)
            ax2.set_title('训练Reward变化曲线', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='lower right', fontsize=10)

            best_epoch_idx = rewards.index(max(rewards))
            best_epoch_num = epochs[best_epoch_idx]
            ax2.axhline(y=best_reward, color='r', linestyle='--', alpha=0.5,
                        label=f'Best: {best_reward:.4f}')
            ax2.scatter([best_epoch_num], [best_reward], color='red',
                        s=100, zorder=5, marker='*')
            ax2.legend(loc='lower right', fontsize=10)

            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            if self.file_manager:
                try:
                    self.file_manager.save_file_record(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        filename=os.path.basename(plot_path),
                        file_type='curve',
                        file_path=plot_path,
                    )
                except Exception as e:
                    logger.warning(f"保存文件记录失败: {e}")
        except Exception as e:
            logger.warning(f"后台绘图线程异常: {e}")

    t = threading.Thread(target=_save, daemon=True)
    t.start()
```

- [ ] **Step 3: 替换 `on_train_epoch_end` 中的绘图代码块**

在 `on_train_epoch_end` 里，找到以下代码块（约 20 行的 try/except）：

```python
        # 生成实时训练曲线图
        try:
            USER_PLOTS_DIR = get_user_plot_dir(self.user_id)
            plot_filename = f"training_curves_{self.session_id[:8]}.png"
            plot_path = os.path.join(USER_PLOTS_DIR, plot_filename)
            
            # 创建包含loss和reward的双子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # 绘制Loss曲线
            ax1.plot(self.history_epochs, self.history_losses, 'b-o', linewidth=2, markersize=6, label='Loss')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('训练Loss变化曲线', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='upper right', fontsize=10)
            
            # 绘制Reward曲线
            ax2.plot(self.history_epochs, self.history_rewards, 'g-o', linewidth=2, markersize=6, label='Reward')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Reward', fontsize=12)
            ax2.set_title('训练Reward变化曲线', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='lower right', fontsize=10)
            
            # 在reward图上标注最佳reward
            best_epoch_idx = self.history_rewards.index(max(self.history_rewards))
            best_epoch_num = self.history_epochs[best_epoch_idx]
            ax2.axhline(y=self.best_reward, color='r', linestyle='--', alpha=0.5, label=f'Best: {self.best_reward:.4f}')
            ax2.scatter([best_epoch_num], [self.best_reward], color='red', s=100, zorder=5, marker='*')
            ax2.legend(loc='lower right', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            # 保存训练曲线文件记录到数据库（file_manager 由 BaseTrainer 注入）
            if self.file_manager:
                try:
                    self.file_manager.save_file_record(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        filename=plot_filename,
                        file_type='curve',
                        file_path=plot_path
                    )
                except Exception as e:
                    logger.warning(f"保存文件记录失败: {e}")
            
            # 通过队列发送图表路径
            self.queue.put(json.dumps({
                'type': 'plot',
                'plot_url': f"/static/model_plots/user_{self.user_id}/{plot_filename}",
                'message': f'Epoch {epoch} 训练曲线已更新'
            }))
        except Exception as e:
            self.queue.put(json.dumps({
                'type': 'warning',
                'message': f'生成训练曲线失败: {str(e)}'
            }))
```

替换为以下代码（复制历史数据 → 发起后台线程 → 立刻推送消息）：

```python
        # 生成实时训练曲线图（后台线程，不阻塞 epoch 边界）
        USER_PLOTS_DIR = get_user_plot_dir(self.user_id)
        plot_filename = f"training_curves_{self.session_id[:8]}.png"
        plot_path = os.path.join(USER_PLOTS_DIR, plot_filename)
        plot_url = f"/static/model_plots/user_{self.user_id}/{plot_filename}"

        # 复制历史列表（后台线程读取，避免与训练线程共享引用）
        self._save_plot_async(
            epochs=list(self.history_epochs),
            losses=list(self.history_losses),
            rewards=list(self.history_rewards),
            best_reward=self.best_reward,
            plot_path=plot_path,
            plot_url=plot_url,
            epoch=epoch,
        )

        # 立刻通知前端（图片由后台线程写入，极短时间内可用）
        self.queue.put(json.dumps({
            'type': 'plot',
            'plot_url': plot_url,
            'message': f'Epoch {epoch} 训练曲线已更新'
        }))
```

- [ ] **Step 4: 验证语法**

```powershell
python -c "from modules.rl_training.base_trainer import ProgressCallback; print('OK')"
```

期望输出：含 `OK`，无 SyntaxError。

- [ ] **Step 5: 整体冒烟测试**

```powershell
python -c "import app; print('app import OK')"
```

期望输出：含 `app import OK`，无 ImportError / AttributeError。

- [ ] **Step 6: Commit**

```bash
git add modules/rl_training/base_trainer.py
git commit -m "perf: async plot generation in ProgressCallback, unblocks epoch boundary"
```

---

## 自我检查

**Spec 覆盖：**
1. ✅ 死代码 `training_functions.py` → Task 1 归档
2. ✅ `TSPTrainer.create_model` 硬编码 REINFORCE → Task 2 修复
3. ✅ `CVRPTrainer.create_model` 无效 override → Task 3 删除
4. ✅ `SDVRPTrainer.create_model` 无效 override → Task 4 删除
5. ✅ 状态写入竞争 → Task 5 改 dict.update()
6. ✅ 每 epoch 同步写图阻塞训练 → Task 6 改后台线程
7. ✅ `FFSPTrainer.create_model` 保留（使用 MatNet，有独立理由）

**未覆盖的 override：** `ffsp_trainer.py` 使用的是 MatNet 而非 REINFORCE，逻辑正确，不修改。

**类型一致性：** `_save_plot_async` 的参数类型（list[int], list[float], float, str, str, int）和 `on_train_epoch_end` 里的调用完全一致。
