import torch.nn as nn
import torch

# --- Diffusion Scheduler  ---
class DiffusionScheduler(nn.Module):
    def __init__(self, num_diffusion_steps: int, schedule_type: str = 'linear'):
        super().__init__()
        if schedule_type not in ['linear', 'cosine']:
            raise ValueError(f"Unsupported schedule_type: {schedule_type}. Choose from 'linear', 'cosine'.")

        self.num_diffusion_steps = num_diffusion_steps
        self.schedule_type = schedule_type

        self.betas = None
        self.alphas = None
        self.alpha_cumprods = None

        self._setup_noise_schedule()

    def _setup_noise_schedule(self):
        if self.schedule_type == 'linear':
            self.betas = torch.linspace(0.0001, 0.02, self.num_diffusion_steps, dtype=torch.float32)
        elif self.schedule_type == 'cosine':
            s = 0.008
            timesteps = torch.arange(self.num_diffusion_steps + 1, dtype=torch.float32)
            alphas_cumprod_cosine = torch.cos(((timesteps / self.num_diffusion_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod_cosine = alphas_cumprod_cosine / alphas_cumprod_cosine[0]
            self.betas = 1 - (alphas_cumprod_cosine[1:] / alphas_cumprod_cosine[:-1])
            self.betas = torch.clip(self.betas, 0.0001, 0.9999)

        self.alphas = 1.0 - self.betas
        self.alpha_cumprods = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_cumprods = torch.sqrt(self.alpha_cumprods)
        self.sqrt_one_minus_alpha_cumprods = torch.sqrt(1.0 - self.alpha_cumprods)

    def add_noise(self, original_latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.long()
        if not ((0 <= timesteps) & (timesteps < self.num_diffusion_steps)).all():
            raise ValueError("Timesteps must be within [0, num_diffusion_steps - 1]")

        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprods[timesteps].view(-1, 1, 1, 1, 1) # Adjust dims
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprods[timesteps].view(-1, 1, 1, 1, 1) # Adjust dims

        noisy_latents = sqrt_alpha_cumprod_t * original_latents + sqrt_one_minus_alpha_cumprod_t * noise
        return noisy_latents

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """
        Performs one denoising step using DDPM sampling equations.
        """
        t = timestep.long().item()
        alpha_prod_t = self.alpha_cumprods[t]
        alpha_prod_t_prev = self.alpha_cumprods[t-1] if t > 0 else torch.tensor(1.0, device=sample.device)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = self.alphas[t]
        current_beta_t = self.betas[t]

        # Current prediction for x_0 (the 'clean' latent)
        pred_original_sample = (sample - current_beta_t.sqrt() * model_output) / current_alpha_t.sqrt()

        # Calculate posterior mean and variance for x_{t-1}
        posterior_variance = current_beta_t * (1. - alpha_prod_t_prev) / (1. - alpha_prod_t)
        posterior_mean = (pred_original_sample * current_alpha_t.sqrt() * (1. - alpha_prod_t_prev) + \
                         sample * alpha_prod_t_prev.sqrt() * current_beta_t) / (1. - alpha_prod_t)

        # Add noise if not the last step (t=0)
        if t > 0:
            noise = torch.randn_like(sample)
            prev_sample = posterior_mean + (posterior_variance.sqrt() * noise)
        else:
            prev_sample = pred_original_sample

        return prev_sample
