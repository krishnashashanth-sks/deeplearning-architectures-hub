import torch.nn as nn
import torch
import torch.nn.functional as F
import math

# --- VAE Components ---
class CustomVAEEncoder(nn.Module):
    def __init__(self, in_channels: int, img_height: int, img_width: int, latent_dim: int):

        super().__init__()
        self.in_channels = in_channels
        self.img_height = img_height
        self.img_width = img_width
        self.latent_dim = latent_dim
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self._final_conv_channels = 128
        self._final_conv_height = img_height // 8
        self._final_conv_width = img_width // 8

        if self._final_conv_height == 0 or self._final_conv_width == 0:
            raise ValueError("Input image dimensions are too small for this encoder architecture.")

        self.fc_mu = nn.Linear(self._final_conv_channels * self._final_conv_height * self._final_conv_width, latent_dim)
        self.fc_logvar = nn.Linear(self._final_conv_channels * self._final_conv_height * self._final_conv_width, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_block(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class CustomVAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, img_height: int, img_width: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.img_height = img_height
        self.img_width = img_width

        self._start_conv_channels = 128
        self._start_conv_height = img_height // 8
        self._start_conv_width = img_width // 8

        if self._start_conv_height == 0 or self._start_conv_width == 0:
            raise ValueError("Input image dimensions are too small for this decoder architecture to match encoder.")

        self.fc_upsample = nn.Linear(latent_dim, self._start_conv_channels * self._start_conv_height * self._start_conv_width)
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(self._start_conv_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # If z comes in as a spatial tensor [B, 4, 8, 8] from the UNet
        if z.dim() == 4:
            # Flatten the spatial dimensions: [B, 4*8*8] -> [B, 256]
            z_flattened = z.view(z.size(0), -1) 
            
            # If the flattened spatial dimensions don't match latent_dim (128)
            if z_flattened.size(1) != self.latent_dim:
                # Project the spatial latents (256) back down to the expected latent_dim (128)
                if not hasattr(self, 'spatial_to_latent_proj'):
                    self.spatial_to_latent_proj = nn.Linear(z_flattened.size(1), self.latent_dim, device=z.device)
                z = self.spatial_to_latent_proj(z_flattened)
            else:
                z = z_flattened

        # Now z is securely shaped as [B, 128], matching your fc_upsample layer
        h = self.fc_upsample(z)
        h = h.view(z.size(0), self._start_conv_channels, self._start_conv_height, self._start_conv_width)
        reconstructed_frames = self.deconv_block(h)

        if reconstructed_frames.shape[-2:] != (self.img_height, self.img_width):
            reconstructed_frames = F.interpolate(reconstructed_frames,
                                                 size=(self.img_height, self.img_width),
                                                 mode='bilinear', align_corners=False)
        return reconstructed_frames


class CustomVAE(nn.Module):
    def __init__(self, in_channels: int, img_height: int, img_width: int, latent_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.img_height = img_height
        self.img_width = img_width
        self.latent_dim = latent_dim
        self.encoder = CustomVAEEncoder(in_channels, img_height, img_width, latent_dim)
        self.decoder = CustomVAEDecoder(latent_dim, in_channels, img_height, img_width)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_frames = self.decoder(z)
        return reconstructed_frames, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# --- Conditioning Blocks (re-defined) ---
class ConceptualTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_attention_heads: int, num_layers: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(512, embedding_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )
        self.linear_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, text_token_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        token_embeddings = self.token_embedding(text_token_ids)
        positions = torch.arange(0, text_token_ids.size(1), device=text_token_ids.device).unsqueeze(0)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        encoded_output = self.transformer_encoder(src=x, src_key_padding_mask=src_key_padding_mask)
        pooled_output = encoded_output[:, 0, :]
        text_embedding = self.linear_projection(pooled_output)
        return text_embedding


class ConceptualMotionModelingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_frames: int, latent_height: int, latent_width: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.conv3d_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=(3, 3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU()
        )
        self.temporal_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, latent_video_frames: torch.Tensor) -> torch.Tensor:
        # Ensure input is (B, F, C, H, W) and permute to (B, C, F, H, W) for 3D Conv
        if latent_video_frames.dim() == 5:
            x = latent_video_frames.permute(0, 2, 1, 3, 4)
        elif latent_video_frames.dim() == 2: # Already an embedding, simulate if used as placeholder
            return latent_video_frames # Return as is if already an embedding
        else:
            raise ValueError("Unexpected input dimensions for MotionModelingBlock")

        motion_features_3d = self.conv3d_block(x)
        return self.temporal_pooling(motion_features_3d).squeeze()


class ConceptualAppearanceModelingBlock(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, latent_height: int, latent_width: int
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.latent_height = latent_height
        self.latent_width = latent_width

        self.conv2d_block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )
        self.spatial_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, latent_image: torch.Tensor) -> torch.Tensor:
        if latent_image.dim() == 4:  # Expected latent image input (B, C, H, W)
            appearance_features_2d = self.conv2d_block(latent_image)
            return (
                self.spatial_pooling(appearance_features_2d)
                .squeeze(-1)
                .squeeze(-1)
            )

        elif (
            latent_image.dim() == 2
        ):  # If a placeholder tensor is passed during pipeline tests
            # Check if the placeholder matches your configured dimensions
            if latent_image.size(1) != self.out_channels:
                # Dynamically project it up to meet the required dimension size (e.g., from 128 to 256)
                projection = nn.Linear(
                    latent_image.size(1),
                    self.out_channels,
                    device=latent_image.device,
                )
                return projection(latent_image)
            return latent_image
        else:
           raise ValueError("Unexpected input dimensions for AppearanceModelingBlock")

# --- SpatioTemporalUNet and its dependencies (re-defined) ---
class TimestepEmbedding(nn.Module):
  def __init__(self,embedding_dim:int):
    super().__init__()
    self.embedding_dim=embedding_dim
    self.linear_1=nn.Linear(embedding_dim,embedding_dim*4)
    self.silu=nn.SiLU()
    self.linear_2=nn.Linear(4*embedding_dim,embedding_dim)
  def forward(self,timesteps:torch.Tensor)->torch.Tensor:
    half_dim=self.embedding_dim//2
    emb=math.log(10000)/(half_dim-1)
    emb=torch.exp(torch.arange(half_dim,device=timesteps.device)*-emb)
    emb=timesteps.float().unsqueeze(1)*emb.unsqueeze(0)
    emb=torch.cat([emb.sin(),emb.cos()],dim=1)
    emb=self.silu(self.linear_1(emb))
    return self.linear_2(emb)

class Downsample3D(nn.Module):
  def __init__(self,in_channels:int,out_channels:int):
    super().__init__()
    self.conv=nn.Conv3d(in_channels,out_channels,kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))
  def forward(self,x:torch.Tensor)->torch.Tensor:
    return self.conv(x)

class Upsample3D(nn.Module):
  def __init__(self,in_channels:int,out_channels:int):
    super().__init__()
    self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.conv(x)

class ConceptualConv3DBlock(nn.Module):
  def __init__(self,in_channels:int,out_channels:int,kernel_size:int=3,stride:int=1,padding:int=1,conditioning_dim:int=0):
    super().__init__()
    self.in_channels=in_channels
    self.out_channels=out_channels
    self.conv_block=nn.Sequential(
        nn.Conv3d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
        nn.GroupNorm(min(32,out_channels),out_channels),
        nn.SiLU(),
        nn.Conv3d(out_channels,out_channels,kernel_size=kernel_size,stride=1,padding=padding),
        nn.GroupNorm(min(32,out_channels),out_channels),
        nn.SiLU()
    )
    self.conditioning_proj=None
    if conditioning_dim >0:
      self.conditioning_proj=nn.Linear(conditioning_dim,out_channels)

  def forward(self,x:torch.Tensor,conditioning_embedding:torch.Tensor=None)->torch.Tensor:
    h=self.conv_block(x)
    if self.conditioning_proj is not None and conditioning_embedding is not None:
      cond_proj=self.conditioning_proj(conditioning_embedding).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
      if cond_proj.shape[2] == 1 and h.shape[2] > 1:
          cond_proj = cond_proj.expand(-1, -1, h.shape[2], -1, -1)
      if cond_proj.shape[3] == 1 and h.shape[3] > 1:
          cond_proj = cond_proj.expand(-1, -1, -1, h.shape[3], -1)
      if cond_proj.shape[4] == 1 and h.shape[4] > 1:
          cond_proj = cond_proj.expand(-1, -1, -1, -1, h.shape[4])
      h=h+cond_proj
    return h


class ConceptualTemporalAttention(nn.Module):
  def __init__(self,feature_dim:int,num_heads:int,conditioning_dim:int=0):
    super().__init__()
    assert feature_dim % num_heads==0,"feature_dim must be divisible by num_heads"
    self.feature_dim=feature_dim
    self.num_heads=num_heads
    self.head_dim=feature_dim//num_heads
    self.query_proj=nn.Linear(feature_dim,feature_dim)
    self.key_proj=nn.Linear(feature_dim,feature_dim)
    self.value_proj=nn.Linear(feature_dim,feature_dim)
    self.output_proj=nn.Linear(feature_dim,feature_dim)
    self.conditioning_proj=None
    if conditioning_dim>0:
      self.conditioning_proj=nn.Linear(conditioning_dim,feature_dim)

  def forward(self,x:torch.Tensor,conditioning_embedding:torch.Tensor=None)->torch.Tensor:
    batch_size,num_frames,_=x.shape
    if self.conditioning_proj is not None and conditioning_embedding is not None:
      cond_proj_expanded = self.conditioning_proj(conditioning_embedding).unsqueeze(1).expand(-1, num_frames, -1)
      x = x + cond_proj_expanded
    queries=self.query_proj(x).view(batch_size,num_frames,self.num_heads,self.head_dim)
    keys=self.key_proj(x).view(batch_size,num_frames,self.num_heads,self.head_dim)
    values=self.value_proj(x).view(batch_size,num_frames,self.num_heads,self.head_dim)

    queries=queries.transpose(1,2)
    keys=keys.transpose(1,2)
    values=values.transpose(1,2)

    attention_scores=torch.matmul(queries,keys.transpose(-2,-1))
    attention_scores=attention_scores/(self.head_dim ** 0.5)
    attention_weights=F.softmax(attention_scores,dim=-1)

    attended_values=torch.matmul(attention_weights,values)
    attended_values=attended_values.transpose(1,2).contiguous().view(batch_size,num_frames,self.feature_dim)

    return self.output_proj(attended_values)


class SpatioTemporalUNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 model_channels: int,
                 num_frames: int,
                 latent_height: int,
                 latent_width: int,
                 text_embedding_dim: int,
                 motion_embedding_dim: int,
                 appearance_embedding_dim: int,
                 num_heads: int = 8,
                 num_res_blocks: int = 2,
                 channel_mults: tuple = (1, 2, 4, 8)
                ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_frames = num_frames
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.num_heads = num_heads
        self.num_res_blocks = num_res_blocks
        self.channel_mults = channel_mults

        self.time_embed_dim = model_channels * 4
        self.conditioning_dim = (
            text_embedding_dim
            + motion_embedding_dim
            + appearance_embedding_dim
            + self.time_embed_dim
        )

        self.time_embed = TimestepEmbedding(self.time_embed_dim)

        self.input_conv = ConceptualConv3DBlock(in_channels, model_channels * channel_mults[0], conditioning_dim=self.conditioning_dim)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        curr_channels = model_channels * channel_mults[0]

        encoder_level_out_channels = []

        for i, mult in enumerate(channel_mults):
            out_channels_for_stage = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(nn.ModuleList([
                    ConceptualConv3DBlock(curr_channels, out_channels_for_stage, conditioning_dim=self.conditioning_dim),
                    ConceptualTemporalAttention(out_channels_for_stage, num_heads, conditioning_dim=self.conditioning_dim)
                ]))
                curr_channels = out_channels_for_stage

            encoder_level_out_channels.append(curr_channels)

            if i < len(channel_mults) - 1:
                self.down_blocks.append(Downsample3D(curr_channels, curr_channels))

        self.mid_block1 = ConceptualConv3DBlock(curr_channels, curr_channels, conditioning_dim=self.conditioning_dim)
        self.mid_attn = ConceptualTemporalAttention(curr_channels, num_heads, conditioning_dim=self.conditioning_dim)
        self.mid_block2 = ConceptualConv3DBlock(curr_channels, curr_channels, conditioning_dim=self.conditioning_dim)

        for i, mult in enumerate(reversed(channel_mults)):
            target_out_channels_for_stage = model_channels * mult

            if i > 0:
                self.up_blocks.append(Upsample3D(curr_channels, target_out_channels_for_stage))
                curr_channels = target_out_channels_for_stage

            skip_channels_from_encoder = encoder_level_out_channels[len(channel_mults) - 1 - i]

            for j in range(self.num_res_blocks):
                input_conv_channels = curr_channels

                if j == 0:
                    input_conv_channels += skip_channels_from_encoder

                self.up_blocks.append(nn.ModuleList([
                    ConceptualConv3DBlock(input_conv_channels, target_out_channels_for_stage, conditioning_dim=self.conditioning_dim),
                    ConceptualTemporalAttention(target_out_channels_for_stage, num_heads, conditioning_dim=self.conditioning_dim)
                ]))
                curr_channels = target_out_channels_for_stage

        self.final_conv = nn.Conv3d(model_channels * channel_mults[0], in_channels, kernel_size=1)

    def forward(self,
                noisy_latent_video: torch.Tensor,
                timestep: torch.Tensor,
                text_embedding: torch.Tensor,
                motion_embedding: torch.Tensor,
                appearance_embedding: torch.Tensor) -> torch.Tensor:

        t_emb = self.time_embed(timestep)

        combined_conditioning = torch.cat(
            [text_embedding, motion_embedding, appearance_embedding, t_emb], dim=-1
        )

        hs = []

        x = noisy_latent_video.permute(0, 2, 1, 3, 4) # (B, F, C, H, W) -> (B, C, F, H, W)

        x = self.input_conv(x, combined_conditioning)

        temp_res_blocks_in_stage = []
        for block_or_downsample in self.down_blocks:
            if isinstance(block_or_downsample, nn.ModuleList):
                conv_block, attn_block = block_or_downsample
                x = conv_block(x, combined_conditioning)

                b_curr, c_curr, f_curr, h_curr, w_curr = x.shape
                x_attn_input = F.adaptive_avg_pool3d(x, output_size=(f_curr, 1, 1)).squeeze(-1).squeeze(-1).permute(0, 2, 1)
                attended_x = attn_block(x_attn_input, combined_conditioning)
                attended_x_expanded = attended_x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(b_curr, c_curr, f_curr, h_curr, w_curr)
                x = x + attended_x_expanded
                temp_res_blocks_in_stage.append(x)

                if len(temp_res_blocks_in_stage) == self.num_res_blocks:
                    hs.append(temp_res_blocks_in_stage[-1])
                    temp_res_blocks_in_stage = []

            elif isinstance(block_or_downsample, Downsample3D):
                x = block_or_downsample(x)

        x = self.mid_block1(x, combined_conditioning)
        b_curr, c_curr, f_curr, h_curr, w_curr = x.shape
        x_attn_input = F.adaptive_avg_pool3d(x, output_size=(f_curr, 1, 1)).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        attended_x = self.mid_attn(x_attn_input, combined_conditioning)
        attended_x_expanded = attended_x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(b_curr, c_curr, f_curr, h_curr, w_curr)
        x = x + attended_x_expanded
        x = self.mid_block2(x, combined_conditioning)

        for block_or_upsample in self.up_blocks:
            if isinstance(block_or_upsample, nn.ModuleList):
                conv_block, attn_block = block_or_upsample

                if conv_block.in_channels > x.shape[1]:
                    skip_feat = hs.pop()

                    if skip_feat.shape[2:] != x.shape[2:]:
                        skip_feat = F.interpolate(skip_feat, size=x.shape[2:], mode='trilinear', align_corners=False)

                    x = torch.cat([x, skip_feat], dim=1)

                x = conv_block(x, combined_conditioning)

                b_curr, c_curr, f_curr, h_curr, w_curr = x.shape
                x_attn_input = F.adaptive_avg_pool3d(x, output_size=(f_curr, 1, 1)).squeeze(-1).squeeze(-1).permute(0, 2, 1)
                attended_x = attn_block(x_attn_input, combined_conditioning)
                attended_x_expanded = attended_x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(b_curr, c_curr, f_curr, h_curr, w_curr)
                x = x + attended_x_expanded

            elif isinstance(block_or_upsample, Upsample3D):
                x = block_or_upsample(x)

        predicted_noise = self.final_conv(x)
        predicted_noise = predicted_noise.permute(0, 2, 1, 3, 4) # (B, C, F, H, W) -> (B, F, C, H, W)

        return predicted_noise
