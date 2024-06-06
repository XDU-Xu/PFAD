import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch as th


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


class GaussianBlur(nn.Module):
    def __init__(self, kernel):
        super(GaussianBlur, self).__init__()
        self.kernel_size = len(kernel)
        # print('kernel size is {0}.'.format(self.kernel_size))
        assert self.kernel_size % 2 == 1, 'kernel size must be odd.'

        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=self.kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0, :, :].unsqueeze_(1)
        padding = self.kernel_size // 2
        x1 = F.conv2d(x1, self.weight, padding=padding)
        return x1


def get_gaussian_blur(kernel_size, device):
    kernel = gkern(kernel_size, 2).astype(np.float32)
    gaussian_blur = GaussianBlur(kernel)
    return gaussian_blur.to(device)


def blur_cond_fn(x, t, x_lr=None, args=None, sample_noisy_x_lr=False, diffusion=None,
                 sample_noisy_x_lr_t_thred=None):

    with th.enable_grad():
        x_in = x.detach().requires_grad_(True)
        if not x_lr is None:
            # x_lr and x_in are of shape BChw, BCHW, they are float type that range from -1 to 1, x_in for small t'

            device_x_in_lr = x_in.device
            blur = get_gaussian_blur(kernel_size=9, device=device_x_in_lr)
            x_in_lr = blur((x_in + 1) / 2)

            if sample_noisy_x_lr:
                t_numpy = t.detach().cpu().numpy()
                spaced_t_steps = [diffusion.timestep_reverse_map[t_step] for t_step in t_numpy]
                if sample_noisy_x_lr_t_thred is None or spaced_t_steps[0] < sample_noisy_x_lr_t_thred:
                    spaced_t_steps = th.Tensor(spaced_t_steps).to(t.device).to(t.dtype)
                    x_lr = diffusion.q_sample(x_lr, spaced_t_steps)

            x_lr = (x_lr + 1) / 2
            mse = (x_in_lr - x_lr) ** 2
            mse = mse.mean(dim=(1, 2, 3))
            mse = mse.sum()
            loss = - mse * args.img_guidance_scale  # move xt toward the gradient direction
        return th.autograd.grad(loss, x_in)[0]
