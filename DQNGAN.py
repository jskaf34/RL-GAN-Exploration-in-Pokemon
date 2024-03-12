import logging
from typing import List, Optional, Type

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from ray.rllib.algorithms.dqn.dqn import (
    DQN
)
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override

from DQNGAN_torch_policy import GANDQNTorchPolicy


logger = logging.getLogger(__name__)



class GAN_DQN(DQN):
    @classmethod
    @override(DQN)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            return GANDQNTorchPolicy
        else:
            return NotImplementedError

