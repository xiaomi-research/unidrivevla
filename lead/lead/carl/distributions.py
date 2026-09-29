# ============================================================================
#  This module is part of the CaRL autonomous driving agent.
#  Original source: https://github.com/autonomousvision/carl
#  All credit for the original implementation goes to the CaRL authors.
# ============================================================================
"""
Probability distributions for sampling actions: Gaussian, Beta, Uniform+Beta.
"""

import torch
from torch import nn
from torch.distributions import Beta, Normal


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution(nn.Module):
    def __init__(self, action_dim: int, dist_init=None, action_dependent_std=False):
        super().__init__()
        assert action_dim == 2

        self.distribution = None
        self.action_dim = action_dim
        self.dist_init = dist_init
        self.action_dependent_std = action_dependent_std

        self.low = None
        self.high = None
        self.log_std_max = 2
        self.log_std_min = -20

        self.suggest_go = nn.Parameter(
            torch.FloatTensor([0.66, -3]),
            requires_grad=False,
        )
        self.suggest_stop = nn.Parameter(
            torch.FloatTensor([-0.66, -3]),
            requires_grad=False,
        )
        self.suggest_turn = nn.Parameter(
            torch.FloatTensor([0.0, -1]),
            requires_grad=False,
        )
        self.suggest_straight = nn.Parameter(
            torch.FloatTensor([3.0, 3.0]),
            requires_grad=False,
        )

    def proba_distribution_net(self, latent_dim: int) -> tuple[nn.Module, nn.Parameter]:
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        if self.action_dependent_std:
            log_std = nn.Linear(latent_dim, self.action_dim)
        else:
            log_std = nn.Parameter(
                -2.0 * torch.ones(self.action_dim),
                requires_grad=True,
            )

        if self.dist_init is not None:
            mean_actions.bias.data[0] = self.dist_init[0][0]
            mean_actions.bias.data[1] = self.dist_init[1][0]
            if self.action_dependent_std:
                log_std.bias.data[0] = self.dist_init[0][1]
                log_std.bias.data[1] = self.dist_init[1][1]
            else:
                init_tensor = torch.FloatTensor(
                    [self.dist_init[0][1], self.dist_init[1][1]],
                )
                log_std = nn.Parameter(init_tensor, requires_grad=True)

        return mean_actions, log_std

    def proba_distribution(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
    ) -> "DiagGaussianDistribution":
        if self.action_dependent_std:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def exploration_loss(self, exploration_suggests) -> torch.Tensor:
        mu = self.distribution.loc.detach().clone()
        sigma = self.distribution.scale.detach().clone()

        for i, suggest_indx in enumerate(exploration_suggests):
            if suggest_indx == 1:
                mu[i, 1] = self.suggest_go[0]
                sigma[i, 1] = self.suggest_go[1]
            elif suggest_indx == 2:
                mu[i, 0] = self.suggest_turn[0]
                sigma[i, 0] = self.suggest_turn[1]
                mu[i, 1] = self.suggest_go[0]
                sigma[i, 1] = self.suggest_go[1]
            elif suggest_indx == 3:
                mu[i, 1] = self.suggest_stop[0]
                sigma[i, 1] = self.suggest_stop[1]

        dist_ent = Normal(mu, sigma)
        exploration_loss = torch.distributions.kl_divergence(
            dist_ent,
            self.distribution,
        )
        return torch.mean(exploration_loss)

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def get_actions(self, sample_type="sample") -> torch.Tensor:
        if sample_type in ("roach", "mean"):
            return self.mode()
        return self.sample()


class BetaDistribution(nn.Module):
    def __init__(self, action_dim=2, dist_init=None):
        super().__init__()
        assert action_dim == 2

        self.distribution = None
        self.action_dim = action_dim
        self.dist_init = dist_init
        self.low = 0.0
        self.high = 1.0

        self.suggest_go = nn.Parameter(
            torch.FloatTensor([2.5, 1.0]),
            requires_grad=False,
        )
        self.suggest_stop = nn.Parameter(
            torch.FloatTensor([1.0, 1.5]),
            requires_grad=False,
        )
        self.suggest_turn = nn.Parameter(
            torch.FloatTensor([1.0, 1.0]),
            requires_grad=False,
        )

    def proba_distribution_net(self, latent_dim: int) -> tuple[nn.Module, nn.Module]:
        linear_alpha = nn.Linear(latent_dim, self.action_dim)
        linear_beta = nn.Linear(latent_dim, self.action_dim)

        if self.dist_init is not None:
            linear_alpha.bias.data[0] = self.dist_init[0][1]
            linear_beta.bias.data[0] = self.dist_init[0][0]
            linear_alpha.bias.data[1] = self.dist_init[1][1]
            linear_beta.bias.data[1] = self.dist_init[1][0]

        alpha = nn.Sequential(linear_alpha)
        beta = nn.Sequential(linear_beta)
        return alpha, beta

    def proba_distribution(self, alpha, beta):
        self.distribution = Beta(alpha, beta)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self):
        return self.distribution.entropy()

    def exploration_loss(self, exploration_suggests) -> torch.Tensor:
        alpha = self.distribution.concentration1.detach().clone()
        beta = self.distribution.concentration0.detach().clone()

        for i, suggest_indx in enumerate(exploration_suggests):
            if suggest_indx == 1:
                alpha[i, 1] = self.suggest_go[0]
                beta[i, 1] = self.suggest_go[1]
            elif suggest_indx == 2:
                alpha[i, 0] = self.suggest_turn[0]
                beta[i, 0] = self.suggest_turn[1]
                alpha[i, 1] = self.suggest_go[0]
                beta[i, 1] = self.suggest_go[1]
            elif suggest_indx == 3:
                alpha[i, 1] = self.suggest_stop[0]
                beta[i, 1] = self.suggest_stop[1]

        dist_ent = Beta(alpha, beta)
        exploration_loss = torch.distributions.kl_divergence(
            self.distribution,
            dist_ent,
        )
        return torch.mean(exploration_loss)

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        alpha = self.distribution.concentration1
        beta = self.distribution.concentration0
        x = torch.zeros_like(alpha)
        x[:, 1] += 0.5
        mask1 = (alpha > 1) & (beta > 1)
        x[mask1] = (alpha[mask1] - 1) / (alpha[mask1] + beta[mask1] - 2)
        mask2 = (alpha <= 1) & (beta > 1)
        x[mask2] = 0.0
        mask3 = (alpha > 1) & (beta <= 1)
        x[mask3] = 1.0
        mask4 = (alpha <= 1) & (beta <= 1)
        x[mask4] = self.distribution.mean[mask4]
        return x

    def evaluate_mean(self) -> torch.Tensor:
        return self.distribution.mean

    def get_actions(self, sample_type="sample") -> torch.Tensor:
        if sample_type == "roach":
            return self.mode()
        elif sample_type == "mean":
            return self.evaluate_mean()
        return self.sample()


class BetaUniformMixtureDistribution(nn.Module):
    def __init__(self, action_dim=2, dist_init=None, uniform_percentage_z=0.1):
        super().__init__()
        assert action_dim == 2

        self.distribution = None
        self.uniform_distribution = None
        self.action_dim = action_dim
        self.dist_init = dist_init
        self.low = 0.0
        self.high = 1.0
        self.beta_perc = 1.0 - uniform_percentage_z
        self.uniform_perc = uniform_percentage_z

        self.suggest_go = nn.Parameter(
            torch.FloatTensor([2.5, 1.0]),
            requires_grad=False,
        )
        self.suggest_stop = nn.Parameter(
            torch.FloatTensor([1.0, 1.5]),
            requires_grad=False,
        )
        self.suggest_turn = nn.Parameter(
            torch.FloatTensor([1.0, 1.0]),
            requires_grad=False,
        )

    def proba_distribution_net(self, latent_dim: int) -> tuple[nn.Module, nn.Module]:
        linear_alpha = nn.Linear(latent_dim, self.action_dim)
        linear_beta = nn.Linear(latent_dim, self.action_dim)

        if self.dist_init is not None:
            linear_alpha.bias.data[0] = self.dist_init[0][1]
            linear_beta.bias.data[0] = self.dist_init[0][0]
            linear_alpha.bias.data[1] = self.dist_init[1][1]
            linear_beta.bias.data[1] = self.dist_init[1][0]

        alpha = nn.Sequential(linear_alpha)
        beta = nn.Sequential(linear_beta)
        return alpha, beta

    def proba_distribution(self, alpha, beta):
        self.action_shape = alpha.shape
        lower_bound = torch.zeros_like(alpha, requires_grad=False, device=alpha.device)
        upper_bound = torch.ones_like(alpha, requires_grad=False, device=alpha.device)
        self.uniform_distribution = torch.distributions.uniform.Uniform(
            lower_bound,
            upper_bound,
        )
        self.distribution = Beta(alpha, beta)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        uniform_pdf = torch.ones_like(
            actions,
            device=actions.device,
            requires_grad=False,
        )
        pdf = (
            self.beta_perc * self.distribution.log_prob(actions).exp()
            + self.uniform_perc * uniform_pdf
        )
        log_prob = torch.log(pdf)
        return sum_independent_dims(log_prob)

    def entropy(self):
        return self.distribution.entropy()

    def exploration_loss(self, exploration_suggests) -> torch.Tensor:
        alpha = self.distribution.concentration1.detach().clone()
        beta = self.distribution.concentration0.detach().clone()

        for i, suggest_indx in enumerate(exploration_suggests):
            if suggest_indx == 1:
                alpha[i, 1] = self.suggest_go[0]
                beta[i, 1] = self.suggest_go[1]
            elif suggest_indx == 2:
                alpha[i, 0] = self.suggest_turn[0]
                beta[i, 0] = self.suggest_turn[1]
                alpha[i, 1] = self.suggest_go[0]
                beta[i, 1] = self.suggest_go[1]
            elif suggest_indx == 3:
                alpha[i, 1] = self.suggest_stop[0]
                beta[i, 1] = self.suggest_stop[1]

        dist_ent = Beta(alpha, beta)
        exploration_loss = torch.distributions.kl_divergence(
            self.distribution,
            dist_ent,
        )
        return torch.mean(exploration_loss)

    def sample(self) -> torch.Tensor:
        prob = torch.rand(1)
        if prob < self.uniform_perc:
            return self.uniform_distribution.rsample()
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        alpha = self.distribution.concentration1
        beta = self.distribution.concentration0
        x = torch.zeros_like(alpha)
        x[:, 1] += 0.5
        mask1 = (alpha > 1) & (beta > 1)
        x[mask1] = (alpha[mask1] - 1) / (alpha[mask1] + beta[mask1] - 2)
        mask2 = (alpha <= 1) & (beta > 1)
        x[mask2] = 0.0
        mask3 = (alpha > 1) & (beta <= 1)
        x[mask3] = 1.0
        mask4 = (alpha <= 1) & (beta <= 1)
        x[mask4] = self.distribution.mean[mask4]
        return x

    def evaluate_mean(self) -> torch.Tensor:
        return self.distribution.mean

    def get_actions(self, sample_type="sample") -> torch.Tensor:
        if sample_type == "roach":
            return self.mode()
        elif sample_type == "mean":
            return self.evaluate_mean()
        return self.sample()
