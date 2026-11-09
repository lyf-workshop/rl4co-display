"""Runtime fixes for RL4CO 0.6 MDAM integration.

The upstream decoder keeps rewards for every decoder path but returns only the
last path's actions. It also exposes unnormalized logits as log-probabilities.
This adapter preserves multi-path tensors for training and returns the best
reward with its matching route for validation, testing, and visualization.
"""

import torch
import torch.nn.functional as F

from rl4co.models.zoo.mdam import MDAMPolicy
from rl4co.models.zoo.mdam.decoder import MDAMDecoder
from rl4co.utils.decoding import decode_logprobs, get_log_likelihood

from .mdam_utils import select_best_mdam_path


_VRP_ENV_NAMES = {
    "cvrp",
    "cvrptw",
    "cvrpmvc",
    "svrp",
    "sdvrp",
    "mtvrp",
}


class RL4CODisplayMDAMDecoder(MDAMDecoder):
    """MDAM decoder with normalized log-probabilities and best-path inference."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_vrp = self.env_name in _VRP_ENV_NAMES

    def _get_logprobs(self, fixed, td, path_index, normalize=True):
        logits, mask = super()._get_logprobs(
            fixed, td, path_index, normalize=False
        )
        if normalize:
            logits = F.log_softmax(logits, dim=-1)
        return logits, mask

    def forward(
        self,
        td,
        encoded_inputs,
        env,
        attn,
        V,
        h_old,
        encoder,
        **decoder_kwargs,
    ):
        decode_type = decoder_kwargs["decode_type"]
        select_best = bool(decoder_kwargs.get("select_best", False))

        first_step_logprobs = []
        for path_index in range(self.num_paths):
            fixed = self._precompute(encoded_inputs.clone(), path_index=path_index)
            logprobs, _ = self._get_logprobs(fixed, td.clone(), path_index)
            first_step_logprobs.append(logprobs[:, 0, :].clamp_min(-1e9))

        if self.num_paths > 1:
            kl_divergences = []
            for left_index in range(self.num_paths):
                for right_index in range(self.num_paths):
                    if left_index == right_index:
                        continue
                    left_logprobs = first_step_logprobs[left_index]
                    right_logprobs = first_step_logprobs[right_index]
                    kl_divergences.append(
                        torch.sum(
                            left_logprobs.exp()
                            * (left_logprobs - right_logprobs),
                            dim=-1,
                        )
                    )
            kl_divergence = torch.stack(kl_divergences, dim=0).mean()
        else:
            kl_divergence = encoded_inputs.new_zeros(())

        rewards_by_path = []
        log_likelihood_by_path = []
        actions_by_path = []

        for path_index in range(self.num_paths):
            path_td = td.clone()
            path_embeddings = encoded_inputs.clone()
            path_attn = attn.clone()
            path_values = V.clone()
            path_old_embeddings = h_old.clone()
            fixed = self._precompute(path_embeddings, path_index=path_index)

            step_logprobs = []
            step_actions = []
            first_mask = None
            previous_mask = None
            step = 0

            while not path_td["done"].all():
                if (
                    step > 1
                    and self.eg_step_gap
                    and step % self.eg_step_gap == 0
                ):
                    mask_for_embedding = previous_mask
                    if not self.is_vrp and first_mask is not None:
                        mask_for_embedding = previous_mask ^ first_mask
                    path_embeddings, _ = encoder.change(
                        path_attn,
                        path_values,
                        path_old_embeddings,
                        mask_for_embedding,
                    )
                    fixed = self._precompute(
                        path_embeddings, path_index=path_index
                    )

                logprobs, mask = self._get_logprobs(
                    fixed, path_td, path_index
                )
                if first_mask is None:
                    first_mask = mask.clone()
                previous_mask = mask

                action = decode_logprobs(
                    logprobs[:, 0, :],
                    mask,
                    decode_type=decode_type,
                )
                path_td.set("action", action)
                path_td = env.step(path_td)["next"]

                step_logprobs.append(logprobs[:, 0, :])
                step_actions.append(action)
                step += 1

            if not step_logprobs:
                raise RuntimeError(
                    "MDAM generated no actions; the environment was already done"
                )

            path_logprobs = torch.stack(step_logprobs, dim=1)
            path_actions = torch.stack(step_actions, dim=1)
            path_reward = env.get_reward(td, path_actions)
            path_log_likelihood = get_log_likelihood(
                path_logprobs, path_actions, mask=None
            )

            rewards_by_path.append(path_reward)
            log_likelihood_by_path.append(path_log_likelihood)
            actions_by_path.append(path_actions)

        rewards = torch.stack(rewards_by_path, dim=1)
        log_likelihood = torch.stack(log_likelihood_by_path, dim=1)

        if select_best:
            rewards, log_likelihood, actions, _ = select_best_mdam_path(
                rewards,
                log_likelihood,
                actions_by_path,
            )
        else:
            # Training consumes all rewards/log-likelihoods. Actions are not
            # part of the MDAM loss, so preserve the historical action shape.
            actions = actions_by_path[-1]

        return rewards, log_likelihood, kl_divergence, actions


class RL4CODisplayMDAMPolicy(MDAMPolicy):
    """Select one coherent MDAM solution outside the training phase."""

    def forward(
        self,
        td,
        env=None,
        phase="train",
        return_actions=True,
        **decoder_kwargs,
    ):
        decoder_kwargs.setdefault("select_best", phase != "train")
        return super().forward(
            td,
            env=env,
            phase=phase,
            return_actions=return_actions,
            **decoder_kwargs,
        )
