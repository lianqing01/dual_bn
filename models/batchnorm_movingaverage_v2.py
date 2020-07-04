import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F
from torch.autograd.function import Function



class BatchNorm_mvv22d(_BatchNorm):
    """
    synchronized batch normalization module extented from ``torch.nn.BatchNormNd``
    with the added stats reduction across multiple processes.
    :class:`apex.parallel.BatchNorm_mvv22d` is designed to work with
    ``DistributedDataParallel``.
    When running in training mode, the layer reduces stats across all processes
    to increase the effective batchsize for normalization layer. This is useful
    in applications where batch size is small on a given process that would
    diminish converged accuracy of the model. The model uses collective
    communication package from ``torch.distributed``.
    When running in evaluation mode, the layer falls back to
    ``torch.nn.functional.batch_norm``.
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``
    Example::
        >>> sbn = apex.parallel.BatchNorm_mvv22d(100).cuda()
        >>> inp = torch.randn(10, 100, 14, 14).cuda()
        >>> out = sbn(inp)
        >>> inp = torch.randn(3, 100, 20).cuda()
        >>> out = sbn(inp)
    """

    warned = False

    def __init__(self, num_features, eps=1e-4, momentum=0.2, affine=True, track_running_stats=True, channel_last=False):
        if channel_last == True:
            raise AttributeError("channel_last is not supported by primitive BatchNorm_mvv22d implementation. Try install apex with `--cuda_ext` if channel_last is desired.")

        if not BatchNorm_mvv22d.warned:
            if hasattr(self, "syncbn_import_error"):
                print("Warning:  using Python fallback for BatchNorm_mvv22d, possibly because apex was installed without --cuda_ext.  The exception raised when attempting to import the cuda backend was: ", self.syncbn_import_error)
            else:
                print("Warning:  using Python fallback for BatchNorm_mvv22d")
            BatchNorm_mvv22d.warned = True

        super(BatchNorm_mvv22d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.register_buffer('running_mu_grad', torch.zeros(num_features))
        self.register_buffer('running_gamma_grad', torch.zeros(num_features))


    @property
    def _momentum(self) -> float:
        if self.num_batches_tracked>=2000:
            return self.momentum
        else:
            return 1.0


    def forward(self, input):
        torch.cuda.nvtx.range_push("sync_bn_fw_with_mean_var")
        mean = None
        var = None
        cast = None
        out = None

        # casting to handle mismatch input type to layer type
        if self.running_mean is not None:
            if self.running_mean.dtype != input.dtype:
                input = input.to(self.running_mean.dtype)
                cast = input.dtype
        elif self.weight is not None:
            if self.weight.dtype != input.dtype:
                input = input.to(self.weight.dtype)
                cast = input.dtype

        if not self.training and self.track_running_stats:
            # fall back to pytorch implementation for inference
            torch.cuda.nvtx.range_pop()
            out = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        else:
            self.num_batches_tracked += 1
            bsz = input.size(0)

            with torch.no_grad():
                channel_first_input = input.transpose(0, 1).contiguous()
                squashed_input_tensor_view = channel_first_input.view(
                    channel_first_input.size(0), -1)
                # total number of data points for each variance entry. Used to calculate unbiased variance estimate
                m = float(squashed_input_tensor_view.size()[1])

                mean = torch.mean(squashed_input_tensor_view, 1)
                sqr_mean = torch.pow(squashed_input_tensor_view, 2).mean(1)

                # var(x) = E (( x - mean_x ) ** 2)
                #        = 1 / N * sum ( x - mean_x ) ** 2
                #        = 1 / N * sum (x**2) - mean_x**2
                var = sqr_mean - mean.pow(2)

                if self.running_mean is not None:
                    self.running_mean = self._momentum * mean + \
                        (1 - self._momentum) * self.running_mean
                if self.running_var is not None:
                    # as noted by the paper, we used unbiased variance estimate of the mini-batch
                    # Var[x] = m / (m-1) * Eb (sample_variance)
                    self.running_var = m / \
                        (m-1) * self._momentum * var + \
                        (1 - self._momentum) * self.running_var

            out = SyncBatchnormFunction.apply(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.running_mu_grad, self.running_gamma_grad, self._momentum)
            torch.cuda.nvtx.range_pop()
        return out.to(cast)
class SyncBatchnormFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_variance, eps, running_mu_grad, running_gamma_grad, momentum):
        torch.cuda.nvtx.range_push("sync_BN_fw")
        # transpose it to channel last to support broadcasting for input with different rank
        c_last_input = input.transpose(1, -1).contiguous().clone()

        ctx.save_for_backward(c_last_input, weight, bias,
                              running_mean, running_variance, running_mu_grad, running_gamma_grad, momentum)
        ctx.eps = eps
        ctx.momentum = momentum

        c_last_input = (c_last_input - running_mean) / \
            torch.sqrt(running_variance + eps)

        if weight is not None:
            c_last_input = c_last_input * weight
        if bias is not None:
            c_last_input = c_last_input + bias

        torch.cuda.nvtx.range_pop()
        return c_last_input.transpose(1, -1).contiguous().clone()
    @staticmethod
    def backward(ctx, grad_output):
        torch.cuda.nvtx.range_push("sync_BN_bw")
        # mini batch mean & var are calculated by forward path.
        # mu = 1./N*np.sum(h, axis = 0)
        # var = 1./N*np.sum((h-mu)**2, axis = 0)
        c_last_input, weight, bias, running_mean, running_variance, running_mu_grad, running_gamma_grad = ctx.saved_tensors

        eps = ctx.eps
        grad_input = grad_weight = grad_bias = None
        num_features = running_mean.size()[0]
        momentum = ctx.momentum

        # transpose it to channel last to support broadcasting for input with different rank
        torch.cuda.nvtx.range_push("carilli field")
        c_last_grad = grad_output.transpose(1, -1).contiguous()
        # squash non-channel dimension so we can easily calculate mean
        c_grad = c_last_grad.view(-1, num_features).contiguous()
        torch.cuda.nvtx.range_pop()

        # calculate grad_input
        if ctx.needs_input_grad[0]:
            # dh = gamma * (var + eps)**(-1. / 2.) * (dy - np.mean(dy, axis=0)
            #     - (h - mu) * (var + eps)**(-1.0) * np.mean(dy * (h - mu), axis=0))
            mean_dy = c_grad.mean(0)
            mean_dy_xmu = (c_last_grad * (c_last_input -
                                          running_mean)).view(-1, num_features).mean(0)
            running_mu_grad = momentum * mean_dy + (1 - momentum) * running_mu_grad
            running_gamma_grad = momentum * mean_dy_xmu + (1 - momentum) * running_gamma_grad
            c_last_grad_input = (c_last_grad - running_mu_grad - (c_last_input - running_mean) / (
                running_variance + eps) * running_gamma_grad) / torch.sqrt(running_variance + eps)
            if weight is not None:
                c_last_grad_input.mul_(weight)
            grad_input = c_last_grad_input.transpose(1, -1).contiguous()

        # calculate grad_weight
        grad_weight = None
        if weight is not None and ctx.needs_input_grad[1]:
            # dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy, axis=0)
            grad_weight = ((c_last_input - running_mean) / torch.sqrt(
                running_variance + eps) * c_last_grad).view(-1, num_features).sum(0)

        # calculate grad_bias
        grad_bias = None
        if bias is not None and ctx.needs_input_grad[2]:
            # dbeta = np.sum(dy, axis=0)
            grad_bias = c_grad.sum(0)

        torch.cuda.nvtx.range_pop()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None
