"""
Trainer for the chip global-routing prototype.

The policy is trained with REINFORCE to output a permutation of two-pin nets.
An A* router evaluates that permutation and provides the terminal reward.
"""

from __future__ import annotations

import copy
from datetime import datetime
import logging
import math
import os
import time
import traceback
from typing import Dict, List, Sequence

import numpy as np
import torch

from modules.envs.chip_routing_env import (
    ChipRoutingGenerator,
    ChipRoutingSolution,
    GridRoutingEnv,
)
from modules.policies.chip_routing_policy import (
    NetOrderingPolicy,
    build_feature_batch,
)

from .base_trainer import BaseTrainer
from .visualizations.chip_routing_viz import (
    create_chip_routing_animation,
    create_chip_routing_comparison,
)
from .visualizations.common import create_training_curve_plot


logger = logging.getLogger("rl4co_display")


class ChipRoutingTrainer(BaseTrainer):
    """REINFORCE trainer for net ordering plus deterministic A* routing."""

    def __init__(
        self,
        config,
        session_id,
        user_id,
        queue,
        training_status,
        get_background_db_func,
        pause_event=None,
    ):
        super().__init__(
            config,
            session_id,
            user_id,
            queue,
            training_status,
            get_background_db_func,
            pause_event,
        )
        self.grid_width = int(config.get("grid_width", 18))
        self.grid_height = int(config.get("grid_height", 18))
        self.num_nets = int(config.get("num_nets", config.get("num_loc", 12)))
        self.num_loc = self.num_nets
        self.obstacle_ratio = float(config.get("obstacle_ratio", 0.10))
        self.edge_capacity = int(config.get("edge_capacity", 1))
        self.congestion_weight = float(config.get("congestion_weight", 2.0))
        self.overflow_weight = float(config.get("overflow_weight", 12.0))
        self.bend_weight = float(config.get("bend_weight", 0.25))
        self.unrouted_penalty = float(config.get("unrouted_penalty", 100.0))
        self.seed = int(config.get("seed", 1234))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        self.effective_batch_size = max(4, min(int(self.batch_size), 24))
        default_batches = math.ceil(self.train_data_size / self.effective_batch_size)
        self.train_batches = max(
            1,
            min(int(config.get("chip_train_batches", default_batches)), 8),
        )
        self.eval_instances = max(1, min(int(config.get("eval_instances", 3)), 3))

        if self.algorithm_name != "reinforce":
            raise ValueError("芯片布线首版仅支持 REINFORCE")
        if self.policy_name not in {"attention", "am"}:
            raise ValueError("芯片布线首版仅支持 Attention 排序策略")

    def initialize_environment(self) -> GridRoutingEnv:
        generator = ChipRoutingGenerator(
            width=self.grid_width,
            height=self.grid_height,
            num_nets=self.num_nets,
            obstacle_ratio=self.obstacle_ratio,
            edge_capacity=self.edge_capacity,
            seed=self.seed,
        )
        return GridRoutingEnv(
            generator=generator,
            congestion_weight=self.congestion_weight,
            overflow_weight=self.overflow_weight,
            bend_weight=self.bend_weight,
            unrouted_penalty=self.unrouted_penalty,
        )

    def _objective_scale(self) -> float:
        return float(max(1, self.num_nets * (self.grid_width + self.grid_height)))

    def _route_orders(
        self,
        env: GridRoutingEnv,
        instances,
        orders: Sequence[Sequence[int]],
    ) -> List[ChipRoutingSolution]:
        return [
            env.route(instance, order)
            for instance, order in zip(instances, orders)
        ]

    def _save_training_curve(
        self,
        epochs: List[int],
        losses: List[float],
        rewards: List[float],
        best_reward: float,
    ) -> str:
        filename = f"training_curves_{self.session_id[:8]}.png"
        path = os.path.join(self.user_plots_dir, filename)
        url = f"/static/model_plots/user_{self.user_id}/{filename}"
        create_training_curve_plot(
            epochs,
            losses,
            rewards,
            path,
            best_reward,
        )
        self.training_status[self.session_id]["plot_url"] = url
        self.send_message(
            "plot",
            f"Epoch {epochs[-1]} 训练曲线已更新",
            plot_url=url,
        )
        return url

    def _handle_controls(self, epoch: int, progress: float) -> bool:
        state = self.training_status.get(self.session_id, {})
        if state.get("stop_requested"):
            state["status"] = "stopped"
            self.send_message(
                "stopped",
                f"训练已中止（已完成 Epoch {epoch}/{self.epochs}）",
                epoch=epoch,
                progress=progress,
            )
            return False

        if self.pause_event is None or self.pause_event.is_set():
            return True

        state["status"] = "paused"
        self.send_message(
            "paused",
            f"训练已暂停（已完成 Epoch {epoch}/{self.epochs}）",
            epoch=epoch,
            progress=progress,
        )
        while not self.pause_event.wait(timeout=0.5):
            if state.get("stop_requested"):
                state["status"] = "stopped"
                self.send_message(
                    "stopped",
                    f"训练已中止（已完成 Epoch {epoch}/{self.epochs}）",
                    epoch=epoch,
                    progress=progress,
                )
                return False
        state["status"] = "running"
        self.send_message("resumed", f"训练已恢复，继续 Epoch {epoch + 1}")
        return True

    def _generate_visualizations(
        self,
        env: GridRoutingEnv,
        initial_policy: NetOrderingPolicy,
        trained_policy: NetOrderingPolicy,
    ) -> Dict:
        self.send_message("info", "开始生成芯片布线对比图、拥塞热力图和动态布线图...")
        instances = [
            env.generate(seed=self.seed + 10_000 + index)
            for index in range(self.eval_instances)
        ]
        features = build_feature_batch(instances, self.device)

        initial_policy.eval()
        trained_policy.eval()
        with torch.no_grad():
            baseline_orders = initial_policy(
                features,
                decode_type="greedy",
            )["order"].cpu().tolist()
            trained_orders = trained_policy(
                features,
                decode_type="greedy",
            )["order"].cpu().tolist()

        baseline_solutions = self._route_orders(
            env,
            instances,
            baseline_orders,
        )
        trained_solutions = self._route_orders(
            env,
            instances,
            trained_orders,
        )
        plot_paths = []
        animation_paths = []

        for index, (instance, baseline, trained) in enumerate(
            zip(instances, baseline_solutions, trained_solutions),
            start=1,
        ):
            plot_filename = (
                f"chip_routing_comparison_{self.session_id[:8]}_{index}.png"
            )
            plot_path = os.path.join(self.user_plots_dir, plot_filename)
            create_chip_routing_comparison(
                instance,
                baseline,
                trained,
                plot_path,
                index=index,
            )
            self._save_file_record(plot_filename, "plot", plot_path)
            plot_paths.append(
                f"/static/model_plots/user_{self.user_id}/{plot_filename}"
            )

            animation_filename = (
                f"chip_routing_animation_{self.session_id[:8]}_{index}.gif"
            )
            animation_path = os.path.join(
                self.user_plots_dir,
                animation_filename,
            )
            try:
                create_chip_routing_animation(
                    instance,
                    trained,
                    animation_path,
                    fps=2,
                )
                self._save_file_record(
                    animation_filename,
                    "animation",
                    animation_path,
                )
                animation_paths.append(
                    f"/static/model_plots/user_{self.user_id}/{animation_filename}"
                )
            except Exception as exc:
                logger.warning("芯片布线动画生成失败: %s", exc, exc_info=True)
                animation_paths.append("")
                self.send_message(
                    "warning",
                    f"第 {index} 组动态布线图生成失败，静态结果仍可查看: {exc}",
                )

        baseline_objective = float(
            np.mean([solution.objective for solution in baseline_solutions])
        )
        trained_objective = float(
            np.mean([solution.objective for solution in trained_solutions])
        )
        metrics = {
            "baseline_objective": round(baseline_objective, 3),
            "trained_objective": round(trained_objective, 3),
            "objective_improvement": round(
                baseline_objective - trained_objective,
                3,
            ),
            "wirelength": round(
                float(
                    np.mean(
                        [solution.wirelength for solution in trained_solutions]
                    )
                ),
                3,
            ),
            "overflow": round(
                float(np.mean([solution.overflow for solution in trained_solutions])),
                3,
            ),
            "routed_ratio": round(
                float(
                    np.mean(
                        [
                            solution.routed_count / self.num_nets
                            for solution in trained_solutions
                        ]
                    )
                ),
                4,
            ),
        }
        self.send_message(
            "info",
            "芯片布线可视化完成: "
            f"训练后平均目标值 {trained_objective:.2f}，"
            f"相对训练前改善 {baseline_objective - trained_objective:+.2f}",
        )
        return {
            "plot_paths": plot_paths,
            "animation_paths": animation_paths,
            "routing_metrics": metrics,
        }

    def _save_checkpoint(
        self,
        policy: NetOrderingPolicy,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
    ) -> None:
        torch.save(
            {
                "problem": "chip_routing",
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": dict(self.config),
            },
            checkpoint_path,
        )
        self._save_file_record(
            os.path.basename(checkpoint_path),
            "checkpoint",
            checkpoint_path,
        )

    def train(self) -> None:
        """Run the standalone REINFORCE loop and emit the standard SSE contract."""
        state = self.training_status.setdefault(self.session_id, {})
        state.update(
            {
                "status": "running",
                "progress": 0,
                "epoch": 0,
                "loss": 0.0,
                "reward": 0.0,
                "best_reward": 0.0,
            }
        )

        try:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            env = self.initialize_environment()
            policy = NetOrderingPolicy(
                embed_dim=int(self.config.get("embed_dim", 128)),
                num_heads=int(self.config.get("num_heads", 4)),
                num_layers=int(self.config.get("num_encoder_layers", 2)),
            ).to(self.device)
            checkpoint_path = os.path.join(
                self.user_checkpoints_dir,
                f"{self.problem_type}-{self.model_type}.ckpt",
            )
            optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=self.learning_rate,
            )

            if (
                self.config.get("resume_checkpoint")
                and os.path.exists(checkpoint_path)
            ):
                try:
                    checkpoint = torch.load(
                        checkpoint_path,
                        map_location=self.device,
                        weights_only=True,
                    )
                except TypeError:
                    checkpoint = torch.load(
                        checkpoint_path,
                        map_location=self.device,
                    )
                policy.load_state_dict(checkpoint["policy_state_dict"])
                optimizer_state = checkpoint.get("optimizer_state_dict")
                if optimizer_state:
                    optimizer.load_state_dict(optimizer_state)
                self.send_message(
                    "info",
                    f"从检查点恢复排序策略: {os.path.basename(checkpoint_path)}",
                )

            initial_policy = copy.deepcopy(policy).to(self.device).eval()

            self.send_message(
                "info",
                "开始训练 Attention Net Ordering 模型，问题类型: CHIP_ROUTING",
            )
            self.send_message(
                "info",
                f"网格={self.grid_width}x{self.grid_height}，网络数={self.num_nets}，"
                f"障碍率={self.obstacle_ratio:.0%}，边容量={self.edge_capacity}",
            )
            self.send_message(
                "info",
                f"使用设备: {self.device}；A* 评估在 CPU 执行，策略梯度在该设备执行",
            )
            if self.effective_batch_size != self.batch_size:
                self.send_message(
                    "info",
                    f"芯片布线包含 CPU A* 搜索，训练批次已从 {self.batch_size} "
                    f"限制为 {self.effective_batch_size}，每轮 {self.train_batches} 个批次",
                )

            history_epochs: List[int] = []
            history_losses: List[float] = []
            history_rewards: List[float] = []
            best_reward = float("-inf")
            curve_url = ""
            objective_scale = self._objective_scale()

            for epoch_index in range(self.epochs):
                completed_epochs = epoch_index
                progress_before = completed_epochs / self.epochs * 100.0
                if not self._handle_controls(completed_epochs, progress_before):
                    return

                policy.train()
                epoch_losses = []
                epoch_rewards = []
                epoch_objectives = []
                epoch_start = time.time()

                for batch_index in range(self.train_batches):
                    if not self._handle_controls(completed_epochs, progress_before):
                        return

                    instances = env.generate_batch(self.effective_batch_size)
                    features = build_feature_batch(instances, self.device)
                    output = policy(features, decode_type="sampling")
                    orders = output["order"].detach().cpu().tolist()
                    solutions = self._route_orders(env, instances, orders)
                    objectives = torch.as_tensor(
                        [solution.objective for solution in solutions],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    rewards = -objectives / objective_scale
                    advantages = rewards - rewards.mean()
                    loss = -(
                        advantages.detach() * output["log_likelihood"]
                    ).mean()
                    loss = loss - self.entropy_coef * output["entropy"].mean()

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()

                    epoch_losses.append(float(loss.detach().cpu()))
                    epoch_rewards.append(float(rewards.mean().detach().cpu()))
                    epoch_objectives.append(float(objectives.mean().detach().cpu()))

                    if batch_index == 0:
                        self.send_message(
                            "info",
                            f"Epoch {epoch_index + 1}/{self.epochs} 训练中 "
                            f"(batch 1/{self.train_batches})",
                        )

                epoch = epoch_index + 1
                loss_value = float(np.mean(epoch_losses))
                reward_value = float(np.mean(epoch_rewards))
                objective_value = float(np.mean(epoch_objectives))
                best_reward = max(best_reward, reward_value)
                progress = epoch / self.epochs * 100.0
                history_epochs.append(epoch)
                history_losses.append(loss_value)
                history_rewards.append(reward_value)
                state.update(
                    {
                        "status": "running",
                        "epoch": epoch,
                        "progress": round(progress, 2),
                        "loss": round(loss_value, 4),
                        "reward": round(reward_value, 4),
                        "best_reward": round(best_reward, 4),
                    }
                )
                curve_url = self._save_training_curve(
                    history_epochs,
                    history_losses,
                    history_rewards,
                    best_reward,
                )
                self.send_message(
                    "progress",
                    "",
                    epoch=epoch,
                    total_epochs=self.epochs,
                    progress=round(progress, 2),
                    loss=round(loss_value, 4),
                    reward=round(reward_value, 4),
                    best_reward=round(best_reward, 4),
                )
                self.send_message(
                    "info",
                    f"Epoch {epoch}/{self.epochs} - Loss: {loss_value:.4f}, "
                    f"Reward: {reward_value:.4f}, Objective: {objective_value:.2f}, "
                    f"耗时: {time.time() - epoch_start:.1f}s",
                )

            self._save_checkpoint(policy, optimizer, checkpoint_path)
            visualization_results = self._generate_visualizations(
                env,
                initial_policy,
                policy,
            )
            state["status"] = "completed"

            if self.bg_session_manager:
                try:
                    self.bg_session_manager.update_session(
                        session_id=self.session_id,
                        status="completed",
                        end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        final_loss=history_losses[-1],
                        final_reward=history_rewards[-1],
                        best_reward=best_reward,
                        checkpoint_path=checkpoint_path,
                    )
                except Exception:
                    logger.exception("记录芯片布线训练完成状态时出错")

            final_results = {
                "model": "attention-net-order",
                "problem": self.problem_type,
                "strategy": "REINFORCE",
                "total_epochs": self.epochs,
                "final_loss": round(history_losses[-1], 4),
                "final_reward": round(history_rewards[-1], 4),
                "best_reward": round(best_reward, 4),
                "training_curve": curve_url,
                "checkpoint_path": checkpoint_path,
                **visualization_results,
            }
            self.send_message("complete", "训练完成！", results=final_results)

        except Exception as exc:
            state["status"] = "error"
            if self.bg_session_manager:
                try:
                    self.bg_session_manager.update_session(
                        session_id=self.session_id,
                        status="failed",
                        end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                except Exception:
                    logger.exception("更新芯片布线训练失败状态时出错")
            full_traceback = traceback.format_exc()
            logger.error(
                "芯片布线训练出错 (session=%s):\n%s",
                self.session_id,
                full_traceback,
            )
            self.send_message(
                "error",
                f"训练出错: {exc}\n\n{full_traceback}",
            )
        finally:
            if self.bg_db:
                try:
                    self.bg_db.close()
                except Exception:
                    logger.warning("关闭芯片布线后台数据库连接失败", exc_info=True)


def train_chip_routing(
    config,
    session_id,
    user_id,
    queue,
    training_status,
    get_background_db_func,
    pause_event=None,
):
    trainer = ChipRoutingTrainer(
        config,
        session_id,
        user_id,
        queue,
        training_status,
        get_background_db_func,
        pause_event,
    )
    trainer.train()


__all__ = ["ChipRoutingTrainer", "train_chip_routing"]

