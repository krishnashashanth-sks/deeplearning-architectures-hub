import torch
import torch.nn as nn
import healpy as hp
import s2fft
import numpy as np

class SphericalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, nside, L_max=None):
        super(SphericalConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nside = nside
        self.L_max = L_max if L_max is not None else 3 * nside

        # Spectral shape for s2fft: (L, 2*L - 1)
        filter_shape = (out_channels, in_channels, self.L_max, 2 * self.L_max - 1)

        # Initialize Complex Weights
        w_real = torch.empty(filter_shape, dtype=torch.float64)
        w_imag = torch.empty(filter_shape, dtype=torch.float64)
        nn.init.xavier_uniform_(w_real)
        nn.init.xavier_uniform_(w_imag)

        self.filters_lm = nn.Parameter(torch.complex(w_real, w_imag))

    def forward(self, x):
        batch_size, _, num_pixels = x.shape
        device = x.device

        # 1. Prepare input (s2fft numpy backend needs numpy arrays)
        # We move to CPU and numpy for the transform
        x_np = x.detach().cpu().to(torch.float64).numpy()
        output_list = []

        for i in range(batch_size):
            input_spectral_list = []

            # --- Spatial -> Spectral ---
            for c_in in range(self.in_channels):
                f_hp = x_np[i, c_in, :]

                # s2fft returns a numpy array
                flm_np = s2fft.forward(
                    f_hp,
                    L=self.L_max,
                    sampling="healpix",
                    method="numpy",
                    nside=self.nside
                )
                # Convert to Torch Tensor for autograd-compatible math
                input_spectral_list.append(torch.from_numpy(flm_np).to(device))

            # Shape: (in_channels, L, 2*L - 1)
            input_spectral = torch.stack(input_spectral_list, dim=0)

            # 2. --- Spectral Convolution (Einsum) ---
            # This happens in PyTorch, preserving the gradient for self.filters_lm
            convolved_flm = torch.einsum('oilm,ilm->olm', self.filters_lm, input_spectral)

            # 3. --- Spectral -> Spatial ---
            out_spatial_list = []
            for c_out in range(self.out_channels):
                # Back to numpy for the inverse transform
                flm_to_inv = convolved_flm[c_out].detach().cpu().numpy()

                f_hp_out = s2fft.inverse(
                    flm_to_inv,
                    L=self.L_max,
                    sampling="healpix",
                    method="numpy",
                    nside=self.nside
                )
                # Convert back to Torch for the final model output
                out_spatial_list.append(torch.from_numpy(f_hp_out).to(device).to(torch.float32))

            output_list.append(torch.stack(out_spatial_list, dim=0))

        return torch.stack(output_list, dim=0)

class SphericalPoolingLayer(nn.Module):
    def __init__(self, nside_in, nside_out):
        super(SphericalPoolingLayer, self).__init__()
        self.nside_in = nside_in
        self.nside_out = nside_out

        if not isinstance(nside_in, int) or not isinstance(nside_out, int) or nside_in < 1 or nside_out < 1:
            raise ValueError("nside_in and nside_out must be positive integers.")

        if nside_in <= nside_out:
            raise ValueError("nside_in must be greater than nside_out for downsampling.")

        if nside_in % nside_out != 0:
            raise ValueError("nside_in must be a multiple of nside_out for proper Healpix downgrading.")

        self.grade_factor = nside_in // nside_out
        self.num_pixels_in = 12 * nside_in**2
        self.num_pixels_out = 12 * nside_out**2

    def forward(self, x):
        # x shape: (batch_size, channels, num_pixels_in)
        batch_size, channels, num_pixels_in = x.shape


        if num_pixels_in != self.num_pixels_in:
            raise ValueError(f"Input pixel count {num_pixels_in} does not match expected {self.num_pixels_in} for nside_in={self.nside_in}.")

        output_pooled_batch = []

        # Process each item in the batch
        for i in range(batch_size):
            pooled_channels = []
            # Process each channel
            for c in range(channels):
                
                input_map_np = x[i, c, :].detach().cpu().numpy().astype(np.float64)

                # Downsample using healpy.ud_grade (summing up the higher res pixels)
                # ud_grade sums the pixels, so we need to divide by the number of pixels summed
                downsampled_map_sum = hp.ud_grade(map_in=input_map_np, nside_out=self.nside_out, order_in='NESTED', order_out='NESTED', pess=False)

                # To get the average, divide by the square of the grading factor
                # Each output pixel is the sum of (grade_factor)^2 input pixels
                pooled_map_np = downsampled_map_sum / (self.grade_factor ** 2)

                # Convert back to torch.Tensor and move to the original device, casting to float32
                pooled_map_tensor = torch.from_numpy(pooled_map_np).to(x.device).to(torch.float32)
                pooled_channels.append(pooled_map_tensor)

            # Stack pooled channel tensors for the current batch item
            output_pooled_batch.append(torch.stack(pooled_channels, dim=0))

        # Stack all batch items to get the final output tensor
        # Final shape: (batch_size, channels, num_pixels_out)
        return torch.stack(output_pooled_batch, dim=0)
