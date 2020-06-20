import torch
import torch.nn as nn

__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]


class BatchRenorm(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-4,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "running_std", torch.ones(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )
        self.weight = torch.nn.Parameter(
            torch.ones(num_features, dtype=torch.float)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float)
        )
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum
        self.num_features = num_features
        self.sample_noise = False

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover
    @property
    def rmax(self) -> torch.Tensor:
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(
            1.0, 2.0
        )

    @property
    def dmax(self) -> torch.Tensor:
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(
            0.0, 2.0
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.transpose(1, -1)
        x = x.reshape(-1,C)
        if self.training:
            batch_mean = x.mean(0)
            batch_std = torch.pow(x, 2).mean(0) - batch_mean.pow(2)
            batch_std = batch_std.clamp(min=self.eps)
            batch_std = torch.sqrt(batch_std)
            r = (
                (batch_std.detach()+self.eps) / (self.running_std.view_as(batch_std) + self.eps
            )).clamp(1/self.rmax, self.rmax)
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean))
                / (self.running_std.view_as(batch_std) + self.eps
            )).clamp(-self.dmax, self.dmax)
            if self.sample_noise is True:
                noise_mean = torch.normal(mean=self.sample_mean, std=self.noise_std_mean).clamp(min=0.1, max=10)
                noise_var = torch.normal(mean=self.sample_mean, std=self.noise_std_var).clamp(min=0.1, max=10)

                r *= noise_var.detach()
                d *= noise_mean.detach()
                r = unsqueeze_tensor(r)
                d = unsqueeze_tensor(d)
                r = r.detach().clamp(1/self.rmax, self.rmax)
                d = d.detach().clamp(-self.dmax, self.dmax)
            else:
                r = unsqueeze_tensor(r)
                d = unsqueeze_tensor(d)
            x = x.view(N, H, W, C)

            x = (x - batch_mean) / (batch_std+self.eps) * r + d
            self.running_mean += self.momentum * (
                batch_mean.detach() - self.running_mean
            )
            self.running_std += self.momentum * (
                batch_std.detach() - self.running_std
            )
            self.num_batches_tracked += 1
        else:

            x = x.view(N, H, W, C)
            x = (x - self.running_mean) / (self.running_std + self.eps)
        if self.affine:
            x = self.weight * x + self.bias
        x = x.transpose(1, -1)
        return x


def unsqueeze_tensor(x, num=3):
    for _ in range(num):
        x = x.unsqueeze(0)
    return x

class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")
