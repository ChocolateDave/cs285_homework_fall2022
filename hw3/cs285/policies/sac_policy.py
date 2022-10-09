
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import sac_utils
from cs285.policies.MLP_policy import MLPPolicy
from torch.distributions.distribution import Distribution


class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20, 2],
                 action_range=[-1, 1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(
            ac_dim,
            ob_dim,
            n_layers,
            size,
            discrete,
            learning_rate,
            training,
            **kwargs
        )

        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(
            np.log(self.init_temperature)
        ).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.learning_rate
        )

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # Formulate entropy term
        return self.log_alpha.exp()

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # Return sample from distribution if sampling
        # if not sampling return the mean of the distribution
        if len(obs.shape) == 1:
            obs = obs[None, ...]

        with torch.no_grad():
            dist = self.forward(ptu.from_numpy(obs))
            if sample:
                action: torch.Tensor = dist.sample()
            else:
                action: torch.Tensor = dist.mean
            # Clamp action range
            action = action.clamp(*self.action_range).cpu().numpy()

        return action

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Distribution:
        # Implement pass through network,
        # computing logprobs and apply correction for Tanh squashing
        # HINT:
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file
        loc: torch.Tensor = self.mean_net(observation)
        scale = torch.exp(
            torch.clamp(
                input=self.logstd,
                min=self.log_std_bounds[0],
                max=self.log_std_bounds[1]
            )
        )
        action_distribution = sac_utils.SquashedNormal(
            loc=loc,
            scale=scale
        )

        return action_distribution

    def update(self, obs, critic):
        # Update actor network and entropy regularizer
        # return losses and alpha value
        self.optimizer.zero_grad()
        self.log_alpha_optimizer.zero_grad()

        policy: Distribution = self.forward(obs)
        action: torch.Tensor = policy.rsample()
        entropy: torch.Tensor = policy.log_prob(action).sum(-1, keepdim=True)
        q_1, q_2 = critic.forward(obs, action)
        min_actor_q = torch.min(q_1, q_2)

        # Policy loss
        actor_loss = (self.alpha.detach() * entropy - min_actor_q).mean()
        actor_loss.backward()
        self.optimizer.step()

        # Alpha loss
        alpha_loss = (
            self.alpha * (-entropy - self.target_entropy).detach()
        ).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, self.alpha
