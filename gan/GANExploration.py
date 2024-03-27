import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree

from typing import Union, Optional
import torch.nn as nn
import torch


from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration, TensorType
from ray.rllib.utils.framework import try_import_tf, try_import_torch, get_variable
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.numpy import convert_to_numpy
from typing import Any
import os 

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

class RandomModel(torch.nn.Module):
    def __init__(self, input_size):
        super(RandomModel, self).__init__()
        self.input_size = input_size
        self.linear = torch.nn.Linear(self.input_size, 1)

    def forward(self, x):
        # flatten the input with respect to the batch size
        x = x.flatten(start_dim=1)
        x = x.float()
        x = self.linear(x)
        # transform to 0/1 thresholding
        x = torch.sigmoid(x)
        x = (x > 0.5)
        return x
    



def load_gan(file_path_for_gan_weights: str, input_size: int):
    #Create a random torch model for testing
    # model = RandomModel(input_size)
    # model.load_state_dict(torch.load(file_path_for_gan_weights))
    discriminator = nn.Sequential(

    nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),

    nn.Flatten(),
    nn.Sigmoid())
    path = os.path.join(r'/Users/jskaf/Documents/Cours ECM 3A/Projet 3A/PROJET3A/weights/gan_game_232.pth')
    #check if the path is correct
    assert os.path.exists(path)
    discriminator.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['discriminator_state_dict'])
    discriminator.train()
    return discriminator



@PublicAPI
class GANExploration(Exploration):
    """Epsilon-greedy Exploration class that produces exploration actions.

    When given a Model's output and a current epsilon value (based on some
    Schedule), it produces a random action (if rand(1) < eps) or
    uses the model-computed one (if rand(1) >= eps).
    """

    def __init__(
        self,
        action_space: gym.spaces.Space,
        *,
        framework: str,
        file_path_for_gan_weights: str,
        input_size: int,
        **kwargs

    ):
        """Create an GANexploration class.

        Args:
            action_space: The action space the exploration should occur in.
            framework: The framework specifier.
            initial_epsilon: The initial epsilon value to use.
            final_epsilon: The final epsilon value to use.
            warmup_timesteps: The timesteps over which to not change epsilon in the
                beginning.
            epsilon_timesteps: The timesteps (additional to `warmup_timesteps`)
                after which epsilon should always be `final_epsilon`.
                E.g.: warmup_timesteps=20k epsilon_timesteps=50k -> After 70k timesteps,
                epsilon will reach its final value.
            epsilon_schedule: An optional Schedule object
                to use (instead of constructing one from the given parameters).
        """
        assert framework is not None
        super().__init__(action_space=action_space, framework=framework, **kwargs)

        # The current timestep value (tf-var or python int).
        self.last_timestep = get_variable(
            np.array(0, np.int64),
            framework=framework,
            tf_name="timestep",
            dtype=np.int64,
        )

        # Build the tf-info-op.
        if self.framework == "tf":
            self._tf_state_op = self.get_state()
        #Load the GAN model with pre-trained weights
        if len(input_size) > 1:
            temp = 1
            for i in input_size:
                temp *= i
            input_size = temp
        self.GAN = load_gan(file_path_for_gan_weights, input_size=input_size)
        #initialize the actions_to_explore with the number of workers
        list_of_booleans = [True]*self.num_workers
        self.actions_to_explore = torch.tensor(list_of_booleans, dtype=torch.bool)


    @override(Exploration)
    def before_compute_actions(
        self,
        *,
        explore: bool,
        timestep: Union[int, TensorType],
        observations: TensorType,
        **kwargs):
        """Implement the action_to_explore : we give to the GAN the batch of observations, and the GAN will tell us which actions to explore"""
        #convert observation in float32
        
        observations = observations.clone().detach().float()


        output = self.GAN(observations.unsqueeze(1))

        #thresholding the output
        output = (output > 0.5)
        #We reshape to have a tensor of size (1, batch_size)
        output = output.view(1, -1).squeeze(0)
        self.actions_to_explore = output

    @override(Exploration)
    def get_exploration_action(
        self,
        *,
        action_distribution: ActionDistribution,
        timestep: Union[int, TensorType],
        explore: Optional[Union[bool, TensorType]] = True,
        #a list of booleans that indicate which actions to explore
    ):

        if self.framework in ["tf2", "tf"]:
            return self._get_tf_exploration_action_op(
                action_distribution, explore, timestep
            )
        else:
            return self._get_torch_exploration_action(
                action_distribution, explore, timestep
            )

    def _get_tf_exploration_action_op(
        self,
        action_distribution: ActionDistribution,
        explore: Union[bool, TensorType],
        timestep: Union[int, TensorType],
    ) -> "tf.Tensor":
        """TF method to produce the tf op for an epsilon exploration action.

        Args:
            action_distribution: The instantiated ActionDistribution object
                to work with when creating exploration actions.

        Returns:
            The tf exploration-action op.
        """
        raise NotImplementedError("TF1 and TF2 are not supported yet.")

    def _get_torch_exploration_action(
        self,
        action_distribution: ActionDistribution,
        explore: bool,
        timestep: Union[int, TensorType],
    ) -> "torch.Tensor":
        """Torch method to produce an epsilon exploration action.

        Args:
            action_distribution: The instantiated
                ActionDistribution object to work with when creating
                exploration actions.

        Returns:
            The exploration-action.
        """
        q_values = action_distribution.inputs
        self.last_timestep = timestep
        exploit_action = action_distribution.deterministic_sample()
        batch_size = q_values.size()[0]
        action_logp = torch.zeros(batch_size, dtype=torch.float)

        # Explore.
        if explore:
            # Get the current epsilon.
            if isinstance(action_distribution, TorchMultiActionDistribution):
                exploit_action = tree.flatten(exploit_action)
                for i in range(batch_size):
                    if self.actions_to_explore[i]:
                        # TODO: (bcahlit) Mask out actions
                        random_action = tree.flatten(self.action_space.sample())
                        for j in range(len(exploit_action)):
                            exploit_action[j][i] = torch.tensor(random_action[j])
                exploit_action = tree.unflatten_as(
                    action_distribution.action_space_struct, exploit_action
                )

                return exploit_action, action_logp

            else:
                # Mask out actions, whose Q-values are -inf, so that we don't
                # even consider them for exploration.
                random_valid_action_logits = torch.where(
                    q_values <= FLOAT_MIN,
                    torch.ones_like(q_values) * 0.0,
                    torch.ones_like(q_values),
                )
                # A random action.
                random_actions = torch.squeeze(
                    torch.multinomial(random_valid_action_logits, 1), axis=1
                )


                # Pick a random action if actions_to_explore is True
                action = torch.where(
                    self.actions_to_explore,
                    random_actions,
                    exploit_action,
                )

                return action, action_logp
        # Return the deterministic "sample" (argmax) over the logits.
        else:
            return exploit_action, action_logp

    @override(Exploration)
    def get_state(self, sess: Optional["tf.Session"] = None):
        if sess:
            return sess.run(self._tf_state_op)
        eps = self.epsilon_schedule(self.last_timestep)
        return {
            "cur_epsilon": convert_to_numpy(eps) if self.framework != "tf" else eps,
            "last_timestep": convert_to_numpy(self.last_timestep)
            if self.framework != "tf"
            else self.last_timestep,
        }

    @override(Exploration)
    def set_state(self, state: dict, sess: Optional["tf.Session"] = None) -> None:
        if self.framework == "tf":
            self.last_timestep.load(state["last_timestep"], session=sess)
        elif isinstance(self.last_timestep, int):
            self.last_timestep = state["last_timestep"]
        else:
            self.last_timestep.assign(state["last_timestep"])
