import torch.nn as nn
import math
import torch

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.nin_shortcut = nn.Identity()

        if temb_channels is not None:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)

        if temb is not None:
            h += self.temb_proj(self.act1(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return self.nin_shortcut(x) + h

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, temb=None):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, temb=None):
        return self.conv(x)

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        # Sinusoidal positional embeddings
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.mlp(embeddings)

# --- Encoder --- 
class Encoder(nn.Module):
    def __init__(self, in_channels, latent_channels, channel_multipliers=[1, 2, 4]):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, latent_channels * channel_multipliers[0], kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        curr_channels = latent_channels * channel_multipliers[0]
        for i in range(len(channel_multipliers) -1):
            next_channels = latent_channels * channel_multipliers[i+1]
            self.down_blocks.append(nn.Sequential(
                ResNetBlock(curr_channels, next_channels),
                Downsample(next_channels)
            ))
            curr_channels = next_channels
        
        self.mid_block = ResNetBlock(curr_channels, curr_channels)
        self.conv_out = nn.Conv2d(curr_channels, latent_channels, kernel_size=3, padding=1) # Output to final latent_dim channels

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.conv_out(x)
        return x

# --- Decoder ---
class Decoder(nn.Module):
    def __init__(self, out_channels, latent_channels, channel_multipliers=[4, 2, 1]):
        super().__init__()
        start_channels = latent_channels * channel_multipliers[0]
        
        self.conv_in = nn.Conv2d(latent_channels, start_channels, kernel_size=3, padding=1)
        self.mid_block = ResNetBlock(start_channels, start_channels)

        self.up_blocks = nn.ModuleList()
        curr_channels = start_channels
        for i in range(len(channel_multipliers) -1):
            next_channels = latent_channels * channel_multipliers[i+1]
            self.up_blocks.append(nn.Sequential(
                ResNetBlock(curr_channels, next_channels),
                Upsample(next_channels)
            ))
            curr_channels = next_channels
        
        self.conv_out = nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.conv_out(x)
        return x

# --- UNetLikeStructure (simplified for latent processing) ---
class UNetLikeStructure(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim, channel_multipliers=[1, 2]):
        super().__init__()
        self.time_embed = TimestepEmbedding(temb_dim)

        self.conv_in = nn.Conv2d(in_channels, in_channels * channel_multipliers[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        curr_channels = in_channels * channel_multipliers[0]
        for i in range(len(channel_multipliers) -1):
            next_channels = in_channels * channel_multipliers[i+1]
            self.down_blocks.append(nn.ModuleList([
                ResNetBlock(curr_channels, next_channels, temb_channels=temb_dim*4),
                Downsample(next_channels)
            ]))
            curr_channels = next_channels

        self.mid_block = ResNetBlock(curr_channels, curr_channels, temb_channels=temb_dim*4)

        self.up_blocks = nn.ModuleList()
        for i in range(len(channel_multipliers) -1, -1, -1):
            next_channels = in_channels * channel_multipliers[i]
            self.up_blocks.append(nn.ModuleList([
                ResNetBlock(curr_channels, next_channels, temb_channels=temb_dim*4),
                Upsample(next_channels) if i > 0 else nn.Identity() # Don't upsample last block
            ]))
            curr_channels = next_channels
        
        self.conv_out = nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1)


    def forward(self, x, timesteps):
        temb = self.time_embed(timesteps)
        
        x = self.conv_in(x)
        h_cache = []

        # Downsampling
        for i, (res_block, down_sample) in enumerate(self.down_blocks):
            x = res_block(x, temb)
            h_cache.append(x)
            x = down_sample(x)
        
        x = self.mid_block(x, temb)

        # Upsampling
        for i, (res_block, up_sample) in enumerate(self.up_blocks):
            if i < len(h_cache): # Skip connection if available
                x = torch.cat([x, h_cache.pop()], dim=1) # Need to adjust channels for skip connection merge
            x = res_block(x, temb)
            x = up_sample(x)
        
        x = self.conv_out(x)
        return x

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, num_channels=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, input):
        return self.main(input)
