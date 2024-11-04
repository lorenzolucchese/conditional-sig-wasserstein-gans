from dataclasses import dataclass

import torch
from torch import optim
from typing import Union, List

from lib.algos.base import BaseAlgo, BaseConfig
from lib.augmentations import SignatureConfig
from lib.augmentations import augment_path_and_compute_signatures, augment_path_and_compute_signature_controls
from lib.utils import sample_indices, to_numpy
from lib.linear_regression import fit_controlled_linear_regression


def sigcwgan_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=2, dim=1).mean()


@dataclass
class SigCWGANConfig:
    mc_size: int
    sig_config_future: SignatureConfig
    sig_config_past: SignatureConfig

    def compute_sig_past(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_past)

    def compute_sig_future(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_future)
    

@dataclass
class MSigCWGANConfig(SigCWGANConfig):
    martingale_indices: Union[List[bool]]

    def compute_sig_control_past(self, x):
        return augment_path_and_compute_signature_controls(x, self.sig_config_past, self.martingale_indices)

    def compute_sig_control_future(self, x):
        return augment_path_and_compute_signature_controls(x, self.sig_config_future, self.martingale_indices)


def calibrate_sigw1_metric(config, x_future, x_past):
    sigs_past = config.compute_sig_past(x_past)
    sigs_future = config.compute_sig_future(x_future)
    assert sigs_past.size(0) == sigs_future.size(0)
    X, Y = to_numpy(sigs_past), to_numpy(sigs_future)
    if type(config) is SigCWGANConfig:
        Z = None
    elif type(config) is MSigCWGANConfig:
        sigs_control_future = config.compute_sig_control_future(x_future)
        Z = to_numpy(sigs_control_future)
    lm = fit_controlled_linear_regression(X, Y, Z)
    sigs_pred = torch.from_numpy(lm.predict(X)).float().to(x_future.device)
    return sigs_pred


def sample_sig_fake(G, q, sig_config, x_past):
    batch_size, p, dim = x_past.shape
    x_past_mc = x_past.repeat(sig_config.mc_size, 1, 1).requires_grad_()                                                            # (batch_size * mc_size, p, dim)
    x_fake = G.sample(q, x_past_mc)                                                                                                 # (batch_size * mc_size, q, dim)
    sigs_fake_future = sig_config.compute_sig_future(x_fake)                                                                        # (batch_size * mc_size, sig_future_dim) where sig_future_dim = dim_aug + ... + dim_aug**depth
    
    X = None
    Y = sigs_fake_future.reshape(sig_config.mc_size, -1)                                                                            # (mc_size, batch_size * sig_future_dim)
    if type(sig_config) is SigCWGANConfig:
        Z = None
    elif type(sig_config) is MSigCWGANConfig:
        sigs_control_fake_future = sig_config.compute_sig_control_future(x_fake)                                                    # (batch_size * mc_size, sig_future_dim)
        Z = sigs_control_fake_future.reshape(sig_config.mc_size, -1)                                                                # (mc_size, batch_size * sig_future_dim)
    sigs_fake_ce = fit_controlled_linear_regression(X, Y, Z, parallel=True).predict(None).reshape(batch_size, -1).float()           # (batch_size, sig_future_dim)
    return sigs_fake_ce, x_fake


class SigCWGAN(BaseAlgo):
    def __init__(
            self,
            base_config: BaseConfig,
            config: SigCWGANConfig,
            x_real: torch.Tensor,
    ):
        super(SigCWGAN, self).__init__(base_config, x_real)
        self.sig_config = config
        self.mc_size = config.mc_size

        self.x_past = x_real[:, :self.p]
        x_future = x_real[:, self.p:]
        self.sigs_pred = calibrate_sigw1_metric(config, x_future, self.x_past)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=1e-2)
        self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, step_size=100, gamma=0.9)

    def sample_batch(self, ):
        random_indices = sample_indices(self.sigs_pred.shape[0], self.batch_size, self.device)  # sample indices
        # sample the least squares signature and the log-rtn condition
        sigs_pred = self.sigs_pred[random_indices].clone().to(self.device)
        x_past = self.x_past[random_indices].clone().to(self.device)
        return sigs_pred, x_past

    def step(self):
        self.G.train()
        self.G_optimizer.zero_grad()  # empty 'cache' of gradients
        sigs_pred, x_past = self.sample_batch()
        sigs_fake_ce, x_fake = sample_sig_fake(self.G, self.q, self.sig_config, x_past)
        loss = sigcwgan_loss(sigs_pred, sigs_fake_ce)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10)
        self.training_loss['loss'].append(loss.item())
        self.training_loss['total_norm'].append(total_norm)
        self.G_optimizer.step()
        self.G_scheduler.step()  # decaying learning rate slowly.
        self.evaluate(x_fake)


class MSigCWGAN(SigCWGAN):
    def __init__(
            self,
            base_config: BaseConfig,
            config: MSigCWGANConfig,
            x_real: torch.Tensor,
    ):
        super(MSigCWGAN, self).__init__(base_config, config, x_real)