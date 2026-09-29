# ============================================================================
#  This module is part of the CaRL autonomous driving agent.
#  Original source: https://github.com/autonomousvision/carl
#  All credit for the original implementation goes to the CaRL authors.
# ============================================================================
"""
PPO policy network for CaRL inference.
Architecture from https://github.com/zhejz/carla-roach
"""

import math

import gymnasium as gym
import numpy as np
import timm
import torch
from torch import nn

from lead.carl.distributions import (
    BetaDistribution,
    BetaUniformMixtureDistribution,
    DiagGaussianDistribution,
)


class CustomCnn(nn.Module):
    def __init__(self, config, n_input_channels):
        super().__init__()
        self.config = config
        self.image_encoder = timm.create_model(
            config.image_encoder,
            in_chans=n_input_channels,
            pretrained=False,
            features_only=True,
        )
        final_width = int(
            self.config.bev_semantics_width
            / self.image_encoder.feature_info.info[-1]["reduction"],
        )
        final_height = int(
            self.config.bev_semantics_height
            / self.image_encoder.feature_info.info[-1]["reduction"],
        )
        final_total_pxiels = final_height * final_width
        self.out_channels = int(1024 / final_total_pxiels)
        self.change_channel = nn.Conv2d(
            self.image_encoder.feature_info.info[-1]["num_chs"],
            self.out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.image_encoder(x)
        x = x[-1]
        x = self.change_channel(x)
        x = torch.flatten(x, start_dim=1)
        return x


class XtMaCNN(nn.Module):
    def __init__(self, observation_space, states_neurons, config):
        super().__init__()
        self.features_dim = config.features_dim
        self.config = config

        n_input_channels = observation_space["bev_semantics"].shape[0]

        if self.config.use_positional_encoding:
            n_input_channels += 2

        if self.config.image_encoder == "roach":
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),
                nn.ReLU(),
            )
        elif self.config.image_encoder == "roach_ln":
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
                nn.LayerNorm((8, 94, 94)),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=5, stride=2),
                nn.LayerNorm((16, 45, 45)),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2),
                nn.LayerNorm((32, 21, 21)),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.LayerNorm((64, 10, 10)),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),
                nn.LayerNorm((128, 4, 4)),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),
                nn.LayerNorm((256, 2, 2)),
                nn.ReLU(),
            )
        elif self.config.image_encoder == "roach_ln2":
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
                nn.LayerNorm((8, 126, 126)),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=5, stride=2),
                nn.LayerNorm((16, 61, 61)),
                nn.ReLU(),
                nn.Conv2d(16, 24, kernel_size=5, stride=2),
                nn.LayerNorm((24, 29, 29)),
                nn.ReLU(),
                nn.Conv2d(24, 32, kernel_size=5, stride=2),
                nn.LayerNorm((32, 13, 13)),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.LayerNorm((64, 6, 6)),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1),
                nn.LayerNorm((128, 4, 4)),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),
                nn.LayerNorm((256, 2, 2)),
                nn.ReLU(),
            )
        else:
            self.cnn = CustomCnn(config, n_input_channels)

        with torch.no_grad():
            sample_bev = torch.as_tensor(
                observation_space["bev_semantics"].sample()[None],
            ).float()
            if self.config.use_positional_encoding:
                x = torch.linspace(-1, 1, self.config.bev_semantics_height)
                y = torch.linspace(-1, 1, self.config.bev_semantics_width)
                y_grid, x_grid = torch.meshgrid(x, y, indexing="ij")
                y_grid = y_grid.to(device=sample_bev.device).unsqueeze(0).unsqueeze(0)
                x_grid = x_grid.to(device=sample_bev.device).unsqueeze(0).unsqueeze(0)
                sample_bev = torch.concatenate((sample_bev, y_grid, x_grid), dim=1)

            self.cnn_out_shape = self.cnn(sample_bev).shape
            self.n_flatten = math.prod(self.cnn_out_shape[1:])

        self.states_neurons = states_neurons[-1]

        if self.config.use_layer_norm:
            self.linear = nn.Sequential(
                nn.Linear(self.n_flatten + states_neurons[-1], 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, config.features_dim),
                nn.LayerNorm(config.features_dim),
                nn.ReLU(),
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(self.n_flatten + states_neurons[-1], 512),
                nn.ReLU(),
                nn.Linear(512, config.features_dim),
                nn.ReLU(),
            )

        states_neurons = [observation_space["measurements"].shape[0]] + list(
            states_neurons,
        )
        self.state_linear = []
        for i in range(len(states_neurons) - 1):
            self.state_linear.append(
                nn.Linear(states_neurons[i], states_neurons[i + 1]),
            )
            if self.config.use_layer_norm:
                self.state_linear.append(nn.LayerNorm(states_neurons[i + 1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

        if self.config.image_encoder in ("roach", "roach_ln", "roach_ln2"):
            self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, bev_semantics, measurements):
        if self.config.use_positional_encoding:
            x = torch.linspace(-1, 1, self.config.bev_semantics_height)
            y = torch.linspace(-1, 1, self.config.bev_semantics_width)
            y_grid, x_grid = torch.meshgrid(x, y, indexing="ij")
            y_grid = (
                y_grid.to(device=bev_semantics.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(bev_semantics.shape[0], -1, -1, -1)
            )
            x_grid = (
                x_grid.to(device=bev_semantics.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(bev_semantics.shape[0], -1, -1, -1)
            )
            bev_semantics = torch.concatenate((bev_semantics, y_grid, x_grid), dim=1)

        x = self.cnn(bev_semantics)
        x = torch.flatten(x, start_dim=1)
        latent_state = self.state_linear(measurements)

        x = torch.cat((x, latent_state), dim=1)
        x = self.linear(x)
        return x


class PPOPolicy(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        policy_head_arch=(256, 256),
        value_head_arch=(256, 256),
        states_neurons=(256, 256),
        config=None,
    ):
        super().__init__()
        self.action_space = action_space
        self.config = config

        self.features_extractor = XtMaCNN(
            observation_space,
            config=config,
            states_neurons=states_neurons,
        )

        if self.config.use_lstm:
            self.lstm = nn.LSTM(
                config.features_dim,
                config.features_dim,
                num_layers=config.num_lstm_layers,
            )
            for name, param in self.lstm.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)

        if self.config.distribution == "beta":
            self.action_dist = BetaDistribution(int(np.prod(action_space.shape)))
        elif self.config.distribution == "normal":
            self.action_dist = DiagGaussianDistribution(
                int(np.prod(action_space.shape)),
                dist_init=self.config.normal_dist_init,
                action_dependent_std=self.config.normal_dist_action_dep_std,
            )
        elif self.config.distribution == "beta_uni_mix":
            self.action_dist = BetaUniformMixtureDistribution(
                int(np.prod(action_space.shape)),
                uniform_percentage_z=self.config.uniform_percentage_z,
            )
        else:
            raise ValueError(
                "Distribution selected that is not implemented. Options: beta, normal, beta_uni_mix",
            )

        self.policy_head_arch = list(policy_head_arch)
        self.value_head_arch = list(value_head_arch)
        self.activation_fn = nn.ReLU

        self.action_space_low = nn.Parameter(
            torch.from_numpy(self.action_space.low),
            requires_grad=False,
        )
        self.action_space_high = nn.Parameter(
            torch.from_numpy(self.action_space.high),
            requires_grad=False,
        )

        self.build()

    def build(self) -> None:
        last_layer_dim_pi = self.features_extractor.features_dim
        policy_net = []
        for layer_size in self.policy_head_arch:
            policy_net.append(nn.Linear(last_layer_dim_pi, layer_size))
            if self.config.use_layer_norm and self.config.use_layer_norm_policy_head:
                policy_net.append(nn.LayerNorm(layer_size))
            policy_net.append(self.activation_fn())
            last_layer_dim_pi = layer_size

        self.policy_head = nn.Sequential(*policy_net)
        self.dist_mu, self.dist_sigma = self.action_dist.proba_distribution_net(
            last_layer_dim_pi,
        )

        if self.config.use_temperature:
            self.temperature_layer = nn.Sequential(
                nn.Linear(last_layer_dim_pi, self.action_dist.action_dim * 2),
                nn.Sigmoid(),
            )

        if self.config.use_value_measurements:
            last_layer_dim_vf = (
                self.features_extractor.features_dim
                + self.config.num_value_measurements
            )
        else:
            last_layer_dim_vf = self.features_extractor.features_dim

        value_net = []
        for layer_size in self.value_head_arch:
            value_net.append(nn.Linear(last_layer_dim_vf, layer_size))
            if self.config.use_layer_norm:
                value_net.append(nn.LayerNorm(layer_size))
            value_net.append(self.activation_fn())
            last_layer_dim_vf = layer_size

        if self.config.use_hl_gauss_value_loss:
            value_net.append(
                nn.Linear(last_layer_dim_vf, self.config.hl_gauss_num_classes),
            )
        else:
            value_net.append(nn.Linear(last_layer_dim_vf, 1))
        self.value_head = nn.Sequential(*value_net)

    def get_features(self, observations) -> torch.Tensor:
        bev_semantics = observations["bev_semantics"].to(dtype=torch.float32)
        measurements = observations["measurements"]
        birdview = bev_semantics / 255.0
        features = self.features_extractor(birdview, measurements)
        return features

    def get_action_dist_from_features(self, features: torch.Tensor, actions=None):
        latent_pi = self.policy_head(features)
        mu = self.dist_mu(latent_pi)
        sigma = self.dist_sigma(latent_pi)

        if actions is not None and self.config.use_rpo:
            z = torch.zeros(mu.shape, dtype=torch.float32, device=mu.device).uniform_(
                -self.config.rpo_alpha,
                self.config.rpo_alpha,
            )
            mu = mu + z

        if self.config.distribution in ("beta", "beta_uni_mix"):
            mu = nn.functional.softplus(mu)
            sigma = nn.functional.softplus(sigma)
            mu = mu + self.config.beta_min_a_b_value
            sigma = sigma + self.config.beta_min_a_b_value

        if self.config.use_temperature:
            temperature = self.temperature_layer(latent_pi)
            mu_temperature = temperature[:, : self.action_dist.action_dim]
            sigma_temperature = temperature[
                :,
                self.action_dist.action_dim : self.action_dist.action_dim * 2,
            ]
            mu_temperature = (
                1.0 - self.config.min_temperature
            ) * mu_temperature + self.config.min_temperature
            sigma_temperature = (
                1.0 - self.config.min_temperature
            ) * sigma_temperature + self.config.min_temperature
            mu = mu / mu_temperature
            sigma = sigma / sigma_temperature

        return (
            self.action_dist.proba_distribution(mu, sigma),
            mu.detach(),
            sigma.detach(),
        )

    def lstm_forward(self, features, lstm_state, done):
        batch_size = lstm_state[0].shape[1]
        hidden = features.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done, strict=True):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def forward(
        self,
        obs_dict: dict[str, np.ndarray],
        actions=None,
        sample_type="sample",
        exploration_suggests=None,
        lstm_state=None,
        done=None,
    ):
        features = self.get_features(obs_dict)

        if self.config.use_lstm:
            features, lstm_state = self.lstm_forward(features, lstm_state, done)

        pred_sem = pred_measure = None

        if self.config.use_value_measurements:
            value_features = torch.cat(
                (features, obs_dict["value_measurements"]),
                dim=1,
            )
        else:
            value_features = features
        values = self.value_head(value_features)
        distribution, mu, sigma = self.get_action_dist_from_features(features, actions)

        if actions is None:
            actions = distribution.get_actions(sample_type)
        else:
            actions = self.scale_action(actions)

        log_prob = distribution.log_prob(actions)

        actions = self.unscale_action(actions)

        entropy = distribution.entropy().sum(1)
        exp_loss = None

        if exploration_suggests is not None:
            exp_loss = distribution.exploration_loss(exploration_suggests)

        return (
            actions,
            log_prob,
            entropy,
            values,
            exp_loss,
            mu,
            sigma,
            distribution.distribution,
            pred_sem,
            pred_measure,
            lstm_state,
        )

    def scale_action(self, action: torch.Tensor, eps=1e-7) -> torch.Tensor:
        d_low, d_high = self.action_dist.low, self.action_dist.high

        if d_low is not None and d_high is not None:
            a_low, a_high = self.action_space_low, self.action_space_high
            action = (action - a_low) / (a_high - a_low) * (d_high - d_low) + d_low
            action = torch.clamp(action, d_low + eps, d_high - eps)
        return action

    def unscale_action(self, action: torch.Tensor) -> torch.Tensor:
        d_low, d_high = self.action_dist.low, self.action_dist.high

        if d_low is not None and d_high is not None:
            a_low, a_high = self.action_space_low, self.action_space_high
            action = (action - d_low) / (d_high - d_low) * (a_high - a_low) + a_low
        return action
