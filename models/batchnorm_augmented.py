import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F
from torch.autograd.function import Function
import torch.nn as nn



class BatchNorm_augmented2d(_BatchNorm):
    """
    synchronized batch normalization module extented from ``torch.nn.BatchNormNd``
    with the added stats reduction across multiple processes.
    :class:`apex.parallel.BatchNorm_augmented2d` is designed to work with
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
        >>> sbn = apex.parallel.BatchNorm_augmented2d(100).cuda()
        >>> inp = torch.randn(10, 100, 14, 14).cuda()
        >>> out = sbn(inp)
        >>> inp = torch.randn(3, 100, 20).cuda()
        >>> out = sbn(inp)
    """

    warned = False

    def __init__(self, num_features, eps=1e-4, momentum=0.1, affine=True, track_running_stats=True, channel_last=False):
        if channel_last == True:
            raise AttributeError("channel_last is not supported by primitive BatchNorm_augmented2d implementation. Try install apex with `--cuda_ext` if channel_last is desired.")

        if not BatchNorm_augmented2d.warned:
            if hasattr(self, "syncbn_import_error"):
                print("Warning:  using Python fallback for BatchNorm_augmented2d, possibly because apex was installed without --cuda_ext.  The exception raised when attempting to import the cuda backend was: ", self.syncbn_import_error)
            else:
                print("Warning:  using Python fallback for BatchNorm_augmented2d")
            BatchNorm_augmented2d.warned = True

        super(BatchNorm_augmented2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.mu_ = nn.Parameter(torch.Tensor(num_features))
        self.gamma_ = nn.Parameter(torch.Tensor(num_features))
        self.alpha_ = nn.Parameter(torch.Tensor(num_features))
        self.alpha1_ = nn.Parameter(torch.Tensor(num_features))
        self.beta_ = nn.Parameter(torch.Tensor(num_features))
        self.beta1_ = nn.Parameter(torch.Tensor(num_features))
        # initialization

        self.mu_.data.fill_(0)
        self.gamma_.data.fill_(1)
        self.alpha_.data.fill_(0)
        self.alpha1_.data.fill_(10)
        self.beta_.data.fill_(0)
        self.beta1_.data.fill_(10)

        #self.register_buffer("mean", torch.zeros(num_features))
        #self.register_buffer("var", torch.zeros(num_features))
        self.register_buffer("tracking_times", torch.tensor(0, dtype=torch.long))


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
            #torch.cuda.nvtx.range_pop()
            #out = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
            input = input.transpose(1, -1).contiguous()
            out = self.weight * (input - self.mu_) / torch.sqrt(self.gamma_ + self.eps) + self.bias
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

            if self.num_batches_tracked <= 2000:
                    self.mu_.data = mean


                    self.gamma_.data = var


            out = self.weight * SyncBatchnormFunction.apply(input, mean, var, self.mu_, self.gamma_, self.alpha_, self.alpha1_, self.beta_, self.beta1_, self.eps, self._momentum) + self.bias
            self.mu_dis = ((mean - self.mu_)**2).mean()
            self.gamma_dis = ((var - self.gamma_)**2).mean()

            torch.cuda.nvtx.range_pop()
        out = out.transpose(1, -1).contiguous()
        return out.to(cast)
class SyncBatchnormFunction(Function):

    @staticmethod
    def forward(ctx, input, mean, var, mu_, gamma_, alpha_, alpha1_, beta_, beta1_, eps, _momentum):
        torch.cuda.nvtx.range_push("sync_BN_fw")
        # transpose it to channel last to support broadcasting for input with different rank
        c_last_input = input.transpose(1, -1).contiguous().clone()
        ctx.eps = eps
        ctx.momentum = _momentum
        r = torch.sqrt(var + eps) / torch.sqrt(gamma_ + eps)
        mu_dis = (mean - mu_)
        gamma_dis = (var - gamma_)
        d = (mu_dis) / torch.sqrt(gamma_+ eps)

        r.clamp_(0.5, 2)
        d.clamp_(-0.5, 0.5)
        ctx.save_for_backward(c_last_input, r, d, mean, var, mu_, gamma_, alpha_, alpha1_, beta_, beta1_)
        c_last_input = (c_last_input - mean) / torch.sqrt(var + eps) * r + d

        torch.cuda.nvtx.range_pop()
        return c_last_input.clone()
    @staticmethod
    def backward(ctx, grad_output):
        torch.cuda.nvtx.range_push("sync_BN_bw")
        # mini batch mean & var are calculated by forward path.
        # mu = 1./N*np.sum(h, axis = 0)
        # var = 1./N*np.sum((h-mu)**2, axis = 0)
        c_last_input, r, d, mean, var, mu_, gamma_, alpha_, alpha1_, beta_, beta1_ = ctx.saved_tensors
        eps = ctx.eps
        momentum = ctx.momentum
        grad_input = grad_mu = grad_gamma_ = None
        grad_alpha_ = grad_alpha1_ = grad_beta_ = grad_beta1_ = None
        num_features = mu_.size(0)

        # transpose it to channel last to support broadcasting for input with different rank
        torch.cuda.nvtx.range_push("carilli field")
        c_last_grad = grad_output
        # squash non-channel dimension so we can easily calculate mean
        c_grad = c_last_grad.contiguous().view(-1, num_features).contiguous()
        torch.cuda.nvtx.range_pop()

        # calculate grad_input
        c_last_input_ = c_last_input.view(-1, num_features).contiguous()
        x_hat = c_last_input_ - mean
        if ctx.needs_input_grad[0]:
            # dh = gamma * (var + eps)**(-1. / 2.) * (dy - np.mean(dy, axis=0)
            #     - (h - mu) * (var + eps)**(-1.0) * np.mean(dy * (h - mu), axis=0))
            mean_dy = c_grad.mean(0)

            mean_dy_xmu = (c_grad * ((x_hat) / torch.sqrt(var + eps)  )).mean(0)
            c_last_grad_input = (c_last_grad - mean_dy -\
                                 ((c_last_input - mean) / torch.sqrt(var + eps) ) *\
                                 mean_dy_xmu  ) / torch.sqrt(var + eps) * r
            grad_input = c_last_grad_input.transpose(1, -1).contiguous()



        # grad for mu_
        x_hat_mean = mean - mu_
        grad_mu_ = -1 * (x_hat_mean) \
        + 1/(alpha1_ + eps) * \
                (c_grad.sum(0) / torch.sqrt((gamma_ + eps)) - alpha_ - 2*beta_*(x_hat_mean))

        grad_alpha_ = -1 * (x_hat_mean)
        grad_alpha1_ = -0.5 * (x_hat_mean**2)


        # grad for alpha
        gamma_dis = var - gamma_

        grad_gamma_ = -1 * gamma_dis\
        + 1/(beta1_ + eps) * (-0.5) *  \
                 (((x_hat) / torch.pow(gamma_+eps, -1.5) * c_grad).sum(0) - beta_)


        # grad for beta
        grad_beta_ = -1 * gamma_dis

        # grad for beta1
        grad_beta1_ = -0.5 * (gamma_dis)**2
        torch.cuda.nvtx.range_pop()
        return grad_input, None, None, grad_mu_, grad_gamma_,\
            grad_alpha_, grad_alpha1_, grad_beta_, grad_beta1_, None, None
