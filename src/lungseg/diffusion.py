"""Forward / reverse diffusion for binary mask refinement."""
import torch


class DiffusionSchedule:
    """Linear-beta DDPM schedule."""

    def __init__(self, timesteps=10, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alpha_bars = self.alpha_bars.to(self.device)
        return self

    def add_noise(self, mask, noise, timestep):
        a_bar = self.alpha_bars[timestep].view(-1, 1, 1, 1, 1)
        return a_bar.sqrt() * mask + (1.0 - a_bar).sqrt() * noise

    @torch.no_grad()
    def sample(self, refiner, image, initial_logits, start=None, inference_steps=None):
        device = image.device
        B = image.shape[0]
        current = torch.randn_like(initial_logits) if start is None else start

        steps = list(reversed(range(self.timesteps)))
        if inference_steps is not None and inference_steps < self.timesteps:
            stride = max(1, self.timesteps // inference_steps)
            steps = list(reversed(range(0, self.timesteps, stride)))

        for step in steps:
            t = torch.full((B,), step, device=device, dtype=torch.long)
            eps_hat = refiner(image, initial_logits, current, t)
            beta = self.betas[step]
            alpha = self.alphas[step]
            a_bar = self.alpha_bars[step]
            mean = (current - ((1.0 - alpha) / (1.0 - a_bar).sqrt()) * eps_hat) / alpha.sqrt()
            if step > 0:
                current = mean + beta.sqrt() * torch.randn_like(current)
            else:
                current = mean
        return current