import logging
import functools
import torch

from torch import nn
from typing import Tuple, Optional

from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import reduce_mean_ignore_inf
from ray.rllib.utils.typing import TensorType, List, Dict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.annotations import override
from ray.rllib.policy.torch_policy import TorchPolicy


logger = logging.getLogger(__name__)

class GANDQNTorchPolicy(DQNTorchPolicy):

    @override(TorchPolicy)
    def compute_actions_from_input_dict(
        self,
        input_dict: Dict[str, TensorType],
        explore: bool = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        with torch.no_grad():
            # Pass lazy (torch) tensor dict to Model as `input_dict`.
            input_dict = self._lazy_tensor_dict(input_dict)
            input_dict.set_training(True)
            # Pack internal state inputs into (separate) list.
            state_batches = [
                input_dict[k] for k in input_dict.keys() if "state_in" in k[:8]
            ]
            # Calculate RNN sequence lengths.
            seq_lens = (
                torch.tensor(
                    [1] * len(state_batches[0]),
                    dtype=torch.long,
                    device=state_batches[0].device,
                )
                if state_batches
                else None
            )
            return self._compute_action_helper(
                input_dict, state_batches, seq_lens, explore, timestep
            )

    @override(TorchPolicy)
    def _compute_action_helper(
        self, input_dict, state_batches, seq_lens, explore, timestep
    ):
        """Shared forward pass logic (w/ and w/o trajectory view API).

        Returns:
            A tuple consisting of a) actions, b) state_out, c) extra_fetches.
        """
        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep
        self._is_recurrent = state_batches is not None and state_batches != []

        # Switch to eval mode.
        if self.model:
            self.model.eval()
        # Call the exploration before_compute_actions hook.
        self.exploration.before_compute_actions(explore=explore, timestep=timestep, observations=input_dict[SampleBatch.CUR_OBS])
        try:
            dist_inputs, dist_class, state_out = self.action_distribution_fn(
                self,
                self.model,
                input_dict=input_dict,
                state_batches=state_batches,
                seq_lens=seq_lens,
                explore=explore,
                timestep=timestep,
                is_training=False,
            )
        # Trying the old way (to stay backward compatible).
        # TODO: Remove in future.
        except TypeError as e:
            if (
                "positional argument" in e.args[0]
                or "unexpected keyword argument" in e.args[0]
            ):
                (
                    dist_inputs,
                    dist_class,
                    state_out,
                ) = self.action_distribution_fn(
                    self,
                    self.model,
                    input_dict[SampleBatch.CUR_OBS],
                    explore=explore,
                    timestep=timestep,
                    is_training=False,
                )
            else:
                raise e
        if not (
            isinstance(dist_class, functools.partial)
            or issubclass(dist_class, TorchDistributionWrapper)
        ):
            raise ValueError(
                "`dist_class` ({}) not a TorchDistributionWrapper "
                "subclass! Make sure your `action_distribution_fn` or "
                "`make_model_and_action_dist` return a correct "
                "distribution class.".format(dist_class.__name__)
            )
        action_dist = dist_class(dist_inputs, self.model)

        # Get the exploration action from the forward results.
        actions, logp = self.exploration.get_exploration_action(
            action_distribution=action_dist, timestep=timestep, explore=explore
        )

        input_dict[SampleBatch.ACTIONS] = actions

        # Add default and custom fetches.
        extra_fetches = self.extra_action_out(
            input_dict, state_batches, self.model, action_dist
        )
        # Action-dist inputs.
        if dist_inputs is not None:
            extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs

        # Action-logp and action-prob.
        if logp is not None:
            extra_fetches[SampleBatch.ACTION_PROB] = torch.exp(logp.float())
            extra_fetches[SampleBatch.ACTION_LOGP] = logp

        # Update our global timestep by the batch size.
        self.global_timestep += len(input_dict[SampleBatch.CUR_OBS])

        return convert_to_numpy((actions, state_out, extra_fetches))