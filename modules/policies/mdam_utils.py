"""Small tensor helpers shared by the MDAM runtime adapter and its tests."""

from collections.abc import Sequence

import torch


def select_best_mdam_path(
    rewards: torch.Tensor,
    log_likelihood: torch.Tensor,
    actions_by_path: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select each batch item's best MDAM reward and its matching action sequence.

    MDAM trains with rewards shaped [batch, num_paths]. During inference the
    application expects one reward and one route per instance, so the path axis
    must be reduced without losing the association between reward and actions.
    Variable-length action sequences are padded with depot action 0.
    """
    if rewards.ndim != 2:
        raise ValueError(f"MDAM rewards must have shape [batch, paths], got {rewards.shape}")
    if log_likelihood.shape != rewards.shape:
        raise ValueError(
            "MDAM log_likelihood must have the same [batch, paths] shape as rewards"
        )

    batch_size, num_paths = rewards.shape
    if len(actions_by_path) != num_paths:
        raise ValueError(
            f"Expected actions for {num_paths} MDAM paths, got {len(actions_by_path)}"
        )
    if not actions_by_path:
        raise ValueError("MDAM must provide at least one action path")

    first_actions = actions_by_path[0]
    if first_actions.ndim != 2 or first_actions.shape[0] != batch_size:
        raise ValueError("Each MDAM action tensor must have shape [batch, steps]")

    best_rewards, best_path_indices = rewards.max(dim=1)
    best_log_likelihood = log_likelihood.gather(
        1, best_path_indices.unsqueeze(1)
    ).squeeze(1)

    max_steps = max(actions.shape[1] for actions in actions_by_path)
    best_actions = first_actions.new_zeros((batch_size, max_steps))

    for path_index, actions in enumerate(actions_by_path):
        if actions.ndim != 2 or actions.shape[0] != batch_size:
            raise ValueError("Each MDAM action tensor must have shape [batch, steps]")
        selected_rows = best_path_indices.to(actions.device).eq(path_index)
        if selected_rows.any():
            best_actions[selected_rows, : actions.shape[1]] = actions[selected_rows]

    return best_rewards, best_log_likelihood, best_actions, best_path_indices
