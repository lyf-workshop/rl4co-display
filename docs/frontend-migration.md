# RL4CO Studio 前端重构（2026）

## 实现边界

继续使用 Flask Blueprint、Jinja2、Vanilla JavaScript、ECharts 和 SSE。训练器、问题环境、策略、算法、认证核心、MySQL schema、Checkpoint 保存规则没有因这次重构被替换。已有未提交的芯片布线、MDAM 和可视化修改保留。

Figma 文件 `0bfdBZJ6cXFSe0O64H1qoC` 的 MCP 返回 UNAUTHORIZED，浏览器访问返回 403。本次依照任务提供的 2026 视觉尺寸、颜色、页面结构和交互要求实现；不能宣称已完成与未授权画布的逐像素比对。

## 前端结构

| 层 | 文件 / 职责 |
| --- | --- |
| 视觉变量 | `static/css/tokens.css`：颜色、字体 fallback、间距、圆角、224px Sidebar |
| 基础与布局 | `base.css`、`layout.css`、`responsive.css`：重置、焦点、Shell、平板抽屉、移动底部导航 |
| 共享组件 | `components.css`：按钮、表单、状态、Toast、Banner、Modal、Tabs、Metric、表格、Loading / Empty / Error |
| 页面样式 | `training.css`、`benchmark.css`、`files.css`、`models.css`、`profile.css`、`auth.css`、`ai-assistant.css` |
| Jinja Shell | `templates/base.html`、`includes/navbar.html`、`includes/icons.html`；认证页面覆盖 Shell blocks |
| 交互基础 | `ui.js`：请求与错误、Loading 按钮、Toast、可访问确认窗口与 focus trap；`app-shell.js`：导航和真实 GPU 摘要 |
| 实验配置 | `index.js`：保留兼容性管理器、条件参数、GPU / 数据集适配；`training-progress.js`：原请求字段构建、真实结果与产物展示 |
| 训练监控 | `training-monitor.js`：启动、唯一 EventSource、状态轮询、心跳、刷新恢复、控制和结果状态 |
| 图表 | `charts.js`：token 主题、ResizeObserver、文字摘要；`static/vendor/echarts.min.js`：本地 ECharts 5.4.3 与许可证 |
| 其他领域 | `benchmark.js`、`file-manager.js`、`model-library.js`、`profile.js`、`auth.js` |
| 助手 | 保留 `ollama-chat-embedded.js` 的代理、历史、配置识别与消息能力；`ai-assistant.js` 适配 Drawer、实际上下文、NDJSON 流和统一反馈 |

旧页面独立 CSS 和旧 navigation.js 经运行时引用审计后移除。历史 `archive/` 和旧开发记录保留。字体不依赖外部 CDN，ECharts 已本地化。

## DOM → 组件 → 请求字段映射

所有原实验配置字段的 ID、name、类型、范围、步长和默认值从原模板迁移；新增的训练 / 验证集大小使用现有 BaseTrainer 配置键。图表、日志等展示 DOM 随新布局更新了全部调用端。

| 原 ID / 域 | 新位置 | 发往后端的字段 |
| --- | --- | --- |
| `problem-select` / `model-select` / `algorithm-select` | 配置卡 | `problem` / `model` / `algorithm` |
| `num-loc` / `epochs` / `batch-size` / `learning-rate` | 配置卡 | `num_loc` / `epochs` / `batch_size` / `learning_rate` |
| `vehicle-capacity` | CVRP / SDVRP / VRPTW 条件区 | `vehicle_capacity` |
| `time-window-width` / `service-time` / `max-time` / `hard-time-windows` | VRPTW 条件区 | `time_window_width` / `service_time` / `max_processing_time` / `hard_time_windows` |
| `num-agents` / `cost-type` | mTSP 条件区 | `num_agents` / `cost_type` |
| `pdp-num-loc` / `force-start-depot` | PDP 条件区 | `num_loc` / `force_start_at_depot` |
| `op-num-loc` / `max-length` / `prize-type` | OP 条件区 | `num_loc` / `max_length` / `prize_type` |
| `pctsp-num-loc` / `penalty-factor` / `prize-required` | PCTSP 条件区 | `num_loc` / `penalty_factor` / `prize_required` |
| `spctsp-num-loc` / `spctsp-penalty-factor` / `spctsp-prize-required` | SPCTSP 条件区 | 同上，按当前问题读取专属 DOM |
| `num-stage` / `num-machine` / `num-job` / `min-time` / `max-time-ffsp` / `flatten-stages` | FFSP 条件区 | `num_stage` / `num_machine` / `num_job` / `min_time` / `max_time` / `flatten_stages` |
| 芯片布线网格、网络、障碍率、边容量 | 芯片布线条件区 | `grid_width` / `grid_height` / `num_nets` / `obstacle_ratio` / `edge_capacity` |
| `num-starts` | POMO 条件区 | `num_starts` |
| `ham-normalization` / `ham-feedforward-hidden` | HAM 条件区 | `normalization` / `feedforward_hidden` |
| `num-augment` / `symnco-num-starts` / `symnco-alpha` / `symnco-beta` | SymNCO 条件区 | `num_augment` / `num_starts` / `symnco_alpha` / `symnco_beta` |
| `n-ants` / `n-iterations-train` / `n-iterations-test` | DeepACO 条件区 | `n_ants` / `n_iterations_train` / `n_iterations_test` |
| `dataset-mode` / `dataset-file` / 选择已有数据集 | 数据来源卡与选择窗口 | `dataset_mode` / `dataset_id`；上传为 multipart `dataset` + `problem_type` |
| GPU radio / `gpu-select-cpu` | 运行资源卡 | `gpu_id`，CPU 为 null；后端 mock GPU 不展示成真实硬件 |
| `resume-checkbox` | 高级参数 | `resume_checkpoint` |
| `train-data-size` / `val-data-size` | 高级参数 | `train_data_size` / `val_data_size`，默认 10000 / 1000 |
| `random-seed` | 配置卡 | 仅芯片布线传 `seed`，现有训练器默认 1234；其他训练器不支持指定种子，控件禁用 |
| `start-button` | 资源侧栏 | `/api/start_training`；校验失败、请求中或已有训练时禁用 |
| `pause-btn` / `resume-btn` / `stop-btn` | 训练监控 | 原 pause / resume / stop API；中止经统一确认 |
| 原进度与结果展示 | monitor / results 独立视图 | SSE 原 `type`、`epoch`、`reward`、`loss`、`results` 等键继续读取 |

仍有少量 `style="display:none"` 和 `style.display`，用于原条件参数契约；颜色、排版和组件样式均放在 CSS。全局 `alert()` / `confirm()` 已从运行时模板和 JS 清除。

## API Inventory 与后端增量

机器可读清单在 `frontend-contract.json`，包括实际 route / method、模板 ID / name / data-*、JS selector / request、核心模块配置键和字典键。用 `python scripts/frontend_audit.py docs/frontend-contract.json` 更新。模块中的教学代码模板含占位符语法，审计记录到 `source_templates`，不将其当成运行时 Python 错误。

| 域 | 保留的接口 |
| --- | --- |
| 训练 | POST `/api/start_training`；GET `/api/training_progress/<session_id>`、`/api/training_status/<session_id>`；POST pause / resume / stop / training_heartbeat 同一路径模式 |
| 兼容性 | `/api/compatibility/info`、`validate`、`constraints/<problem>`、`available_policies`、`available_algorithms`、`recommend` |
| GPU | `/api/gpu_status`、`/api/gpu_allocate`、`/api/gpu_release` |
| 数据集与文件 | upload_dataset、list_datasets、delete_dataset、list_files、delete_file、download_checkpoint、delete_session、delete_by_session、clear_all_files |
| 账户 | `/api/register`、`/api/login`、`/api/logout`、`/api/current_user`；原 GET 页面 / 重定向保留 |
| 统计 | 原 `/api/user_stats`、`/api/user_activity` 等清单内接口 |
| 助手 | `/api/ollama/tags`、`/api/ollama/chat`、`/api/llm/providers`、`/api/llm/chat`，继续使用原请求与 NDJSON 响应 |

后端只在三个文件增加 UI 所需能力：

- `app_pages.py`：模型数据库 ID 映射实际策略，详情展示兼容性矩阵与可用算法；研究方法仍可阅读，不伪装成已经接入的可训练策略。
- `app_files.py`：数据集列表附带真实字节数；增加登录用户目录内的 `/api/download_dataset/<dataset_id>`；`list_files` 附加 `all_files`，保留原 `files` 字段，修复同类型多个产物只显示一份的问题。
- `app_training.py`：附加 UI metadata 和当前用户 `/api/training_sessions`，合并 DB 历史与进程内实时状态；status 响应保持原字段并附加 context / total_epochs；SSE / 控制加入 Session 所有者校验，修复旧接口的跨账户访问漏洞；终态结果在流发送前留存，也可从未消费队列读回而不消耗事件。训练分派、线程、暂停事件、GPU 分配 / 释放、训练器和数据库 schema 沿用原逻辑。

`live` 表示 Session 是否仍有当前服务器进程的状态；避免将进程重启后 DB 中遗留的 running 记录误当成可恢复任务。浏览器刷新能恢复仍在运行的后台 Session，服务器进程重启不会重新启动训练。

训练模式来自本次运行的后端信息。历史数据库没有训练模式字段时显示“模式未记录”，不把旧记录推断为真实训练或模拟训练。

## 页面和数据来源

- 工作台：12 种原有问题、8 种策略、5 个算法选项与全部专项参数；配置、监控、结果为同一路由中的状态。
- 算法对比：账户内完成的 Session，可按问题、规模 / 数据集 / 模式、多实验与指标筛选；保留原项目论文基准表到 `static/data/benchmark-reference.json`，作为独立数据源并标注出处属性，不混同账户训练记录。
- 文件与数据：数据集、训练结果、Checkpoint 三个 Tab，实际数量 / 大小，上传 / 下载 / 使用 / 删除；Session 删除准确说明后端保留会话记录。
- 模型：MODEL_DATABASE 的全部 23 条记录；列表搜索 / 分类，通用详情、兼容性与“用于新实验”。
- 账户：后端真实用户名、邮箱、创建 / 登录时间、统计和活动。没有制造无 API 的编辑按钮。
- 登录 / 注册：原认证 API、session 和重定向；字段错误、密码不一致、Loading / disabled。
- 助手：440px 右侧 Drawer；实际服务 / 模型、用户作用域历史、实验上下文、快捷 prompt、流式消息、复制、重新生成、停止生成、重试和应用建议。配置应用不会自动启动任务。

缺失指标显示 `—` 或隐藏。Best Epoch 仅在后端提供该字段时出现。Checkpoint 的共享文件名规则和静态产物路径仍遵循原后端，不在 UI 发明新文件或新的存储保证。

## SSE 生命周期与可访问性

启动请求只从表单提交发送。重连、心跳、状态轮询和刷新恢复均不调用 start_training。每页保留一个 EventSource，切换离开 / 终态时关闭；网络异常显示重连 Banner，浏览器自动重连并辅以状态快照轮询和手动重试。in-flight status 结果不能覆盖已经结束或已切换的 Session。

启动后 URL 保留 `?session=<id>`。活动任务可从用户作用域缓存恢复，结果页可通过 URL 再次打开；恢复时读取服务端配置并还原表单、数据集、GPU 和高级参数，不提交新的训练请求。服务器内存状态被清理或服务重启后，界面回到配置，历史实验仍可在算法对比与文件页读取。

确认窗口支持 aria-modal、标题 / 描述关联、Tab focus trap、Esc / 遮罩关闭、返回触发元素、请求中禁止重复提交 / 关闭、失败留窗、成功更新列表。助手与数据集窗口使用同一焦点层管理，确认窗口位于 Drawer 上层。

响应式断点：1200px 以上 224px Sidebar；768–1199px 导航抽屉；767px 以下单列表单、双列指标、5 项底部导航、全屏助手。表格使用局部横向滚动。focus-visible、label、图标按钮名称、曲线文字摘要和 reduced-motion 已实现。

## 验证入口

```powershell
conda run -n rl4co-display python -m pytest tests/ -q
python scripts/frontend_audit.py docs/frontend-contract.json
git diff --check
```

启动真实 Flask、登录独立验收账户后运行只读布局检查：

```powershell
npx --yes --package @playwright/cli playwright-cli -s=rl4co-review open http://127.0.0.1:5000
# 在该浏览器中登录后：
npx --yes --package @playwright/cli playwright-cli -s=rl4co-review run-code --filename scripts/verify_frontend_browser.js
```

该脚本验证四个宽度、六个业务页面、12 种问题的条件区、兼容性、模型预选和 Drawer，并在 `output/playwright/` 写截图。它是 CLI function expression；末尾不能加独立分号。真实写入操作的验收使用独立账户，测试数据仅存在该账户目录，不替换生产接口。

本次实际执行结果与环境限制见 [frontend-verification.md](frontend-verification.md)。完整工作区状态与 tracked diff 统计分别保存在 `frontend-git-status.txt`、`frontend-git-diff-stat.txt`；其中包含开始任务前已有的用户修改，不能把全部统计归因于此次重构。
