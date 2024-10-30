"""
Simple augmentations to enhance the capability of capturing important features in the first components of the
signature.
"""
from dataclasses import dataclass
from typing import List, Tuple
import warnings

import signatory
import torch

__all__ = ['AddLags', 'Concat', 'Cumsum', 'LeadLag', 'Scale']


def get_time_vector(size: int, length: int) -> torch.Tensor:
    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


def lead_lag_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate paths.
    """
    x_rep = torch.repeat_interleave(x, repeats=2, dim=1)
    x_ll = torch.cat([x_rep[:, :-1], x_rep[:, 1:]], dim=2)
    return x_ll


def lead_lag_transform_with_time(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate paths.
    """
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    t_rep = torch.repeat_interleave(t, repeats=3, dim=1)
    x_rep = torch.repeat_interleave(x, repeats=3, dim=1)
    x_ll = torch.cat([
        t_rep[:, 0:-2],
        x_rep[:, 1:-1],
        x_rep[:, 2:],
    ], dim=2)
    return x_ll


def cat_lags(x: torch.Tensor, m: int) -> torch.Tensor:
    q = x.shape[1]
    assert q >= m, 'Lift cannot be performed. q < m : (%s < %s)' % (q, m)
    x_lifted = list()
    for i in range(m):
        x_lifted.append(x[:, i:i + (q - m + 1)])
    return torch.cat(x_lifted, dim=-1)


@dataclass
class BaseAugmentation:
    pass

    def apply(self, *args: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('Needs to be implemented by child.')
    
    def martingale_indices_after_augmentation(self, *args: List[List[bool]]) -> List[bool]:
        raise NotImplementedError('Needs to be implemented by child.')


@dataclass
class Scale(BaseAugmentation):
    scale: float = 1

    def apply(self, x: torch.Tensor):
        return self.scale * x
    
    def martingale_indices_after_augmentation(self, indices: List[bool]) -> List[bool]:
        return indices


@dataclass
class Concat(BaseAugmentation):

    @staticmethod
    def apply(x: torch.Tensor, y: torch.Tensor):
        return torch.cat([x, y], dim=-1)
    
    def martingale_indices_after_augmentation(self, indices_x: List[bool], indices_y: List[bool]) -> List[bool]:
        return indices_x + indices_y


@dataclass
class Cumsum(BaseAugmentation):
    dim: int = 1

    def apply(self, x: torch.Tensor):
        return x.cumsum(dim=self.dim)
    
    def martingale_indices_after_augmentation(self, indices: List[bool]) -> List[bool]:
        # this assumes input is martinagle difference sequence (e.g. log returns) 
        # and thus output is martingale (e.g. log-prices) 
        if any(indices): 
            warnings.warn('Using Cumsum() with martingale_indices, ensure input is martingale difference.')
        return indices


@dataclass
class AddLags(BaseAugmentation):
    m: int = 2

    def apply(self, x: torch.Tensor):
        return cat_lags(x, self.m)
    
    def martingale_indices_after_augmentation(self, indices: List[bool]) -> List[bool]:
        # only (subset) of last d entries (the leaders) remain martingales
        return [False] * len(indices) * (self.m - 1) + indices


@dataclass
class LeadLag(BaseAugmentation):
    with_time: bool = False

    def apply(self, x: torch.Tensor):
        if self.with_time:
            return lead_lag_transform_with_time(x)
        else:
            return lead_lag_transform(x)
        
    def martingale_indices_after_augmentation(self, indices: List[bool]) -> List[bool]:
        # only (subset) of last d entries (the leaders) remain martingales
        if self.with_time:
            return [False] + [False] * len(indices) + indices
        else:
            return [False] * len(indices) + indices


def _apply_augmentation(x: torch.Tensor, y: torch.Tensor, augmentation) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    if type(augmentation).__name__ == 'Concat':  # todo
        return y, augmentation.apply(x, y)
    else:
        return y, augmentation.apply(y)


def apply_augmentations(x: torch.Tensor, augmentations: Tuple[BaseAugmentation, ...]) -> torch.Tensor:
    y = x
    for augmentation in augmentations:
        x, y = _apply_augmentation(x, y, augmentation)
    return y


@dataclass
class SignatureConfig:
    augmentations: Tuple[BaseAugmentation, ...]
    depth: int
    basepoint: bool = False


def augment_path_and_compute_signatures(x: torch.Tensor, config: SignatureConfig) -> torch.Tensor:
    y = apply_augmentations(x, config.augmentations)
    return signatory.signature(y, config.depth, basepoint=config.basepoint)


def get_standard_augmentation(scale: float) -> Tuple[BaseAugmentation, ...]:
    return tuple([Scale(scale), Cumsum(), Concat(), AddLags(m=2), LeadLag(with_time=False)])


def augment_path_and_compute_signature_controls(x: torch.Tensor, config: SignatureConfig, martingale_indices: List[bool]) -> torch.Tensor:
    y = apply_augmentations(x, config.augmentations)
    batch, length, channels = y.shape
    if not any(martingale_indices):
        return torch.zeros((batch, sum([channels**i for i in range(1, config.depth + 1)])))
    else:
        martingale_indices_after_augmentations = get_martingale_indices_after_augmentations(martingale_indices, config.augmentations)
        signature_control_indices = torch.tensor(get_signature_control_indices(config.depth, martingale_indices_after_augmentations))
        signatures_stream = signatory.signature(y, config.depth, stream=True)                                               # shape: (batch, length - 1, channels + ... + channels**depth)
        signatures_lower = signatures_stream[:, :-1, :-channels**config.depth]                                              # shape: (batch, length - 2, channels + ... + channels**(depth-1))
        # pre-pend signature starting values at zero
        signatures_start = torch.cat([torch.zeros(batch, 1, channels**i) for i in range(1, config.depth)], dim=2)           # shape: (batch, 1, channels + ... + channels**(depth-1))	
        signatures_lower = torch.cat([signatures_start, signatures_lower], dim=1)                                           # shape: (batch, length - 1, channels + ... + channels**(depth-1))        
        # append 0-th order signature
        signatures_lower = torch.cat([torch.ones(batch, length - 1, 1), signatures_lower], dim=2)                           # shape: (batch, length - 1, 1 + channels + ... + channels**(depth-1))
        # integrate
        corrections = torch.einsum('ijk,ijl->ikl', signatures_lower, y.diff(axis=1)).flatten(start_dim=1)                   # shape: (batch, channels + ... + channels**depth)
        corrections[:, ~signature_control_indices] = 0.                                                                     # shape: (batch, channels + ... + channels**depth)
        return corrections


def _get_martingale_indices_after_augmentation(indices_x: List[bool], indices_y: List[bool], augmentation: BaseAugmentation) -> Tuple[List[bool], List[bool]]:
    if type(augmentation).__name__ == 'Concat':  # todo
        return indices_y, augmentation.martingale_indices_after_augmentation(indices_x, indices_y)
    else:
        return indices_y, augmentation.martingale_indices_after_augmentation(indices_y)


def get_martingale_indices_after_augmentations(martingale_indices_x: List[bool], augmentations: Tuple[BaseAugmentation, ...]):
    martingale_indices_y = martingale_indices_x
    for augmentation in augmentations:
        martingale_indices_x, martingale_indices_y = _get_martingale_indices_after_augmentation(martingale_indices_x, martingale_indices_y, augmentation)
    return martingale_indices_y


def get_signature_control_indices(depth: int, martingale_indices: List[bool]) -> List[bool]:
    channels = len(martingale_indices)
    sig_indices = []
    for i in range(1, depth + 1):
        # in each signature level, of size channels**i, we want to set to True indices of signature levels ending in martigale indices
        for j in martingale_indices:
            if j:
                sig_indices.extend([True] * channels**(i-1))
            else:
                sig_indices.extend([False] * channels**(i-1))
    return sig_indices
