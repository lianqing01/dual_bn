from __future__ import division

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.autograd import Function

class LagrangianFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        # input shape: [1, C, 1, 1]
        # weight shape: [1, C, 1, 1]
        # output shape: [1, C, 1, 1]
        ctx.save_for_backward(input, weight)
        output = input * weight
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * weight
        if ctx.needs_input_grad[1]:
            # gradient ascent
            grad_weight = -1 * grad_output * input
        return grad_input, grad_weight


class Constraint_Norm(nn.Module):

    def __init__(self, num_features, weight_decay=1e-3, get_optimal_lagrangian=False, pre_affine=True, post_affine=True):
        super(Constraint_Norm, self).__init__()
        self.num_features = num_features
        self.pre_affine=pre_affine
        self.post_affine = post_affine
        self.set_dim()
        self.mu_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))
        self.gamma_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))

        #initialization
        self.mu_.data.fill_(0)
        self.gamma_.data.fill_(1)
        self.lagrangian = Constraint_Lagrangian(num_features,
                                                weight_decay=weight_decay,
                                                get_optimal_lagrangian=get_optimal_lagrangian)

        # strore mean and variance for reference
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.zeros(num_features))
        self.register_buffer("real_mu", torch.zeros(num_features))
        self.register_buffer("real_gamma", torch.zeros(num_features))

        self.register_buffer("tracking_times", torch.tensor(0, dtype=torch.long))
        self.update_affine_only = False
        self.sample_noise=False
        self.noise_data_dependent=False
        self.sample_mean = None
        self.noise_mu_ = []
        self.add_noise = None
        self.noise_gamma_ = []
        self.summarize_x_hat = []
        self.summarize_x_hat_noise = []
        self.eps = 1e-4

    def store_norm_stat(self):
        self.noise_mu_.append(self.mu_.grad.clone().detach())
        self.noise_gamma_.append(self.gamma_.grad.clone().detach())

    def summarize_norm_stat(self):
        self.noise_mu_ = torch.stack(self.noise_mu_)
        self.noise_mu_sum = self.noise_mu_.clone()
        self.noise_mu_ = self.mu_.detach() - self.noise_mu_
        self.noise_gamma_ = torch.stack(self.noise_gamma_)
        self.nosie_gamma_sum = self.noise_gamma_.clone()
        self.noise_gamma_ = self.gamma_.detach() - self.noise_gamma_


        self.noise_mu_mean = torch.mean(self.noise_mu_, dim=0)
        self.noise_gamma_mean = torch.mean(self.noise_gamma_, dim=0)
        self.noise_mu_var = torch.var(self.noise_mu_, dim=0).clamp(min=0)
        self.noise_mu_std = torch.sqrt(self.noise_mu_var) * self.lambda_noise_weight
        #self.noise_gamma_var = torch.var(1 / (self.noise_gamma_**2+1e-5), dim=0).clamp(min=0)
        self.noise_gamma_var = torch.var(self.noise_gamma_**2, dim=0).clamp(min=0)
        self.noise_gamma_std = torch.sqrt(self.noise_gamma_var) * self.lambda_noise_weight



        self.noise_mu_ = []
        self.noise_gamma_ = []



    def get_mean_var(self):
        with torch.no_grad():
            mean = self.mean / (self.tracking_times + 1e-4)
            var = self.var / (self.tracking_times + 1e-4)
            mean = mean.abs().mean()
            var = var.abs().mean()
            var = self.var / (self.tracking_times + 1e-4)
            mean = mean.abs().mean()
            var = var.abs().mean()
        return mean, var

    def get_real_mean_var(self):
         with torch.no_grad():
            mean = self.real_mu / (self.tracking_times)
            var = self.real_gamma / (self.tracking_times)
         return mean, var


    def set_dim(self):
        raise NotImplementedError

    def _initialize_mu(self, with_affine=False):
        self.mean = self.mean / self.tracking_times
        if with_affine:
            self.old_mu_ = self.mu_

        self.mu_.data += self.mean.view(self.mu_.size())

    def _initialize_gamma(self, with_affine=False):
        if with_affine:
            self.old_gamma_ = self.gamma_
        self.var = self.var / self.tracking_times
        self.var -= 1
        self.gamma_.data = torch.sqrt((self.var.view(self.gamma_.size())+1) * self.gamma_**2).data

    def _initialize_affine(self):
        #temp = self.post_affine_layer.u_.data / (self.old_gamma_.data + self.eps)
        #self.post_affine_layer.u_.data.copy_(temp * self.gamma_.data))


        #self.post_affine_layer.c_.data -= (temp -temp1)
        del self.old_mu_
        del self.old_gamma_


    def forward(self, x):

        # mean
        with torch.no_grad():
            self.real_mu += x.mean(dim=self.norm_dim)
        x_hat = (x - self.mu_) / torch.sqrt(self.gamma_**2 + self.eps)
        if self.sample_noise and self.training:
                noise_mean = torch.normal(mean=self.sample_mean.fill_(1), std=self.sample_mean_std)
                noise_mean = noise_mean.view(self.mu_.size()).clamp(min=0.1, max=10)
                x = x - (self.mu_ * noise_mean.detach() )
        else:
            x = x - self.mu_
        # var
        with torch.no_grad():
            self.real_gamma += torch.sqrt((x**2).clamp(min=0)).mean(dim=self.norm_dim)
        if self.sample_noise and self.training:
                noise_var = torch.normal(mean=self.sample_mean.fill_(1), std=self.sample_var_std)
                noise_var = noise_var.view(self.gamma_.size()).clamp(min=0.1, max=10).detach()

                x = x / torch.sqrt((self.gamma_ * noise_var)**2 + self.eps)
        else:

            x = x / torch.sqrt(self.gamma_**2 + self.eps)

        mean = self.lagrangian.get_weighted_mean(x_hat, self.norm_dim)
        self.mean += mean.detach()
        var = self.lagrangian.get_weighted_var(x_hat, self.gamma_, self.norm_dim)
        self.var += var.detach()

        self.tracking_times += 1
        #self.summarize_x_hat.append(x.detach())
        if self.post_affine != False:
            x = self.post_affine_layer(x)
        return x



    def reset_norm_statistics(self):
        self.mean.fill_(0)
        self.var.fill_(0)
        self.tracking_times.fill_(0)
        self.real_mu.fill_(0)
        self.real_gamma.fill_(0)



class Constraint_Norm1d(Constraint_Norm):
    def __init__(self, num_features, pre_affine=True, post_affine=True):
        super(Constraint_Norm1d, self).__init__(num_features, pre_affine=pre_affine, post_affine=post_affine)

    def set_dim(self):
        self.feature_dim = [1, self.num_features]
        self.norm_dim = [0]
        if self.post_affine != False:
            self.post_affine_layer = Constraint_Affine1d(self.num_features)

class Constraint_Norm2d(Constraint_Norm):
    def __init__(self, num_features, pre_affine=True, post_affine=True):
        super(Constraint_Norm2d, self).__init__(num_features, pre_affine=pre_affine, post_affine=post_affine)

    def set_dim(self):
        self.feature_dim = [1, self.num_features, 1, 1]
        self.norm_dim = [0, 2, 3]
        if self.post_affine != False:
            self.post_affine_layer = Constraint_Affine2d(self.num_features)




class Constraint_Lagrangian(nn.Module):

    def __init__(self, num_features, weight_decay=1e-4, get_optimal_lagrangian=False):
        super(Constraint_Lagrangian, self).__init__()
        self.num_features = num_features
        self.lambda_ = nn.Parameter(torch.Tensor(num_features))

        self.xi_ = nn.Parameter(torch.Tensor(num_features))
        self.lambda_.data.fill_(0)
        self.xi_.data.fill_(0)
        self.weight_decay = weight_decay
        self.get_optimal_lagrangian = get_optimal_lagrangian

    def get_weighted_mean(self, x, norm_dim):
        mean = x.mean(dim=norm_dim)
        self.weight_mean = LagrangianFunction.apply(mean, self.xi_)
        self.weight_mean = self.weight_mean.mean()
        return mean


    def get_weighted_var(self, x,  gamma, norm_dim):
        var = x**2 - 1
        var = var.mean(dim=norm_dim)
        self.weight_var = LagrangianFunction.apply(var, self.lambda_)
        self.weight_var = self.weight_var.mean()
        return var+1
    def get_weight_mean_var(self):
        return (self.weight_mean, self.weight_var)

    def get_weight_mean_var_abs(self):
        return (self.weight_mean_abs, self.weight_var_abs)


class Constraint_Affine(nn.Module):
    def __init__(self, num_features):
        super(Constraint_Affine, self).__init__()
        self.num_features = num_features
        self.set_dim()

        self.c_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))
        self.u_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))
        self.c_.data.fill_(0)
        self.u_.data.fill_(1)

    def set_dim(self):
        raise NotImplementedError


    def forward(self, x):
        return x * self.u_ + self.c_

class Constraint_Affine1d(Constraint_Affine):
    def __init__(self, num_features):
        super(Constraint_Affine1d, self).__init__(num_features)

    def set_dim(self):
        self.feature_dim = [1, self.num_features]


class Constraint_Affine2d(Constraint_Affine):
    def __init__(self, num_features):
        super(Constraint_Affine2d, self).__init__(num_features)

    def set_dim(self):
        self.feature_dim = [1, self.num_features, 1, 1]


