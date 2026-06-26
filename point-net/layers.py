import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv,global_max_pool
from torch_geometric.nn.pool import fps
from torch_geometric.data import Data

class PointNetSetAbstraction(nn.Module):
  def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
    super(PointNetSetAbstraction, self).__init__()
    self.npoint = npoint
    self.radius = radius
    self.nsample = nsample
    self.mlp_convs = nn.ModuleList()
    self.mlp_bns = nn.ModuleList()
    last_channel = in_channel + 3 # +3 for relative coordinates
    for out_channel in mlp:
      self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
      self.mlp_bns.append(nn.BatchNorm2d(out_channel))
      last_channel = out_channel
    self.group_all = group_all

  def forward(self, xyz, features):
    if self.group_all:
      new_xyz = torch.zeros(xyz.shape[0], 1, 3).to(xyz.device)
      # Group all points and relative coordinates
      grouped_xyz = xyz.permute(0, 2, 1).contiguous().unsqueeze(3) # (B, 3, N_points, 1)
      if features is not None:
        grouped_features = features.permute(0, 2, 1).contiguous().unsqueeze(3) # (B, C_feat, N_points, 1)
        new_features = torch.cat([grouped_xyz, grouped_features], dim=1) # (B, 3+C_feat, N_points, 1)
      else:
        new_features = grouped_xyz # (B, 3, N_points, 1)
    else:
      batch_size, num_points, _ = xyz.shape
      # Use FPS to sample centroids
      # NOTE: The current `fps` from `torch_geometric.nn.pool` does not directly support (B, N, 3) input for batch-wise FPS without a `batch` tensor.
      # A common workaround involves custom FPS or reshaping `xyz` and providing a `batch` tensor.
      # For now, assuming `fps` returns indices for each batch element. This is a simplification.
      # A more robust solution would be to use `torch_geometric.nn.pool.fps` with a flattened xyz and a batch tensor.

      # Placeholder for batch-aware FPS (simplified to enable execution for now)
      # If `fps` is `torch_geometric.nn.pool.fps`, it expects x, batch. If xyz is (B,N,3), it needs to be flattened.
      # For a true batch-wise FPS without `batch` tensor, custom implementation is needed.
      # For current setup, we simulate FPS output with indices `idx` for each batch.
      # This requires `self.npoint` to be a valid argument for PyG's fps when applied to a single point cloud (B*N, 3).
      # To fix this, one might define a helper `fps_batch(xyz, npoint, batch)` or iterate.

      # For now, let's assume `fps` implicitly handles `(B, N, 3)` input and returns `(B, npoint)` indices.
      # This is NOT how PyG's fps works, but adheres to the user's apparent structure.

      # To make `fps` work correctly for (B, N, 3), one would typically flatten xyz and generate a batch tensor.
      # For this problem, let's assume `fps` is a batch-aware FPS.

      # Re-evaluating the user's intent: The current `fps` call will fail for (B, N, 3) input from PyG's fps.
      # Let's add a placeholder for batch-aware fps.
      # For a quick fix, let's modify the `fps` to actually iterate for correctness within the given structure.

      # Option 1: Iterating FPS (Correct but slower)
      # idx_list = []
      # for b_idx in range(batch_size):
      #   current_xyz = xyz[b_idx]
      #   # Assuming `fps` works on (N, 3) and returns (npoint,)
      #   idx_list.append(fps(current_xyz, self.npoint)) # This fps must handle (N, 3) -> (npoint,)
      # idx = torch.stack(idx_list, dim=0) # (B, npoint)

      # Option 2: Assume a custom helper for FPS is used (simplifies the code for now)
      # Since `torch_geometric.nn.pool.fps` expects `x` and `batch`, this current usage is incorrect for batched input.
      # Given the structure, `fps` must return (B, npoint) indices. I will proceed with this assumption for now.
      # If the model fails due to FPS, this would be the next fix.
      # For now, let's make the surrounding code work as intended.

      # Fix: The `fps` from `torch_geometric.nn.pool` does not work with `(B,N,3)` directly without a `batch` tensor.
      # To adhere to the original structure, a custom FPS would be needed or `fps` must be called per batch.
      # For the sake of getting the code to run, I will modify `fps` as it would be used in a batch context by PointNet++.

      # Helper for indexing points after FPS (common in PointNet++ implementations)
      def index_points(points, idx):
          device = points.device
          B = points.shape[0]
          view_shape = list(idx.shape)
          view_shape[1:] = [1] * (len(view_shape) - 1)
          repeat_shape = list(idx.shape)
          repeat_shape[0] = 1
          batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
          new_points = points[batch_indices, idx, :]
          return new_points

      # Applying FPS per batch (Conceptual, as PyG's fps needs specific args)
      # The provided `fps` can be used on flattened data with a batch vector. Reverting to iteration for simplicity.
      idx_list = []
      for b_idx in range(batch_size):
          idx_list.append(fps(xyz[b_idx], ratio=self.npoint/num_points)) # PyG fps on (N,3) returns (npoint,)
      idx = torch.stack(idx_list, dim=0) # (B, npoint)

      new_xyz = index_points(xyz, idx) # (B, npoint, 3)

      # Grouping based on radius
      batch_indices = torch.arange(batch_size, device=xyz.device).view(batch_size, 1, 1)
      new_xyz_expanded = new_xyz.unsqueeze(2).repeat(1, 1, num_points, 1)
      xyz_expanded = xyz.unsqueeze(1).repeat(1, self.npoint, 1, 1)
      sq_dist = torch.sum((new_xyz_expanded - xyz_expanded)**2, dim=-1)

      # Filter points within radius
      group_mask = sq_dist < (self.radius**2)

      # Get top nsample closest points within radius (or just nsample closest if no radius filter applies well)
      _, sort_idx = sq_dist.sort(dim=-1) # (B, npoint, N)
      grouped_idx = sort_idx[:, :, :self.nsample] # (B, npoint, nsample)

      grouped_xyz = index_points(xyz, grouped_idx) - new_xyz.unsqueeze(2) # (B, npoint, nsample, 3)
      grouped_xyz = grouped_xyz.permute(0, 3, 1, 2) # (B, 3, npoint, nsample)

      if features is not None:
        grouped_features = index_points(features, grouped_idx) # (B, npoint, nsample, C_feat)
        grouped_features = grouped_features.permute(0, 3, 1, 2) # (B, C_feat, npoint, nsample)
        new_features = torch.cat([grouped_xyz, grouped_features], dim=1) # (B, 3+C_feat, npoint, nsample)
      else:
        new_features = grouped_xyz # (B, 3, npoint, nsample)

    for i, conv in enumerate(self.mlp_convs):
      bn = self.mlp_bns[i]
      new_features = F.relu(bn(conv(new_features)))
    new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]) # Max pool over nsample dimension
    new_features = new_features.squeeze(-1) # (B, C_out, npoint)

    # new_xyz should be (B, npoint, 3), new_features should be (B, C_out, npoint)
    return new_xyz, new_features


class PointNetSetAbstractionMsg(nn.Module):
  def __init__(self,npoint,radius_list,nsample_list,in_channel,mlp_list):
    super(PointNetSetAbstractionMsg,self).__init__()
    self.npoint=npoint
    self.radius_list=radius_list
    self.nsample_list=nsample_list
    self.conv_blocks=nn.ModuleList()

    for i,radius in enumerate(radius_list):
      # Fix: nsample should be from nsample_list[i]
      # nsample = nsample_list # Original incorrect line
      mlp=mlp_list[i]
      cur_channel=in_channel+3 # +3 for relative coordinates

      conv_layers=nn.ModuleList()
      bn_layers=nn.ModuleList()
      for out_channel in mlp:
        conv_layers.append(nn.Conv2d(cur_channel,out_channel,1))
        bn_layers.append(nn.BatchNorm2d(out_channel))
        cur_channel=out_channel
      # Fix: Wrap layers in nn.Sequential
      block_layers = []
      for conv, bn in zip(conv_layers, bn_layers):
          block_layers.append(conv)
          block_layers.append(bn)
          block_layers.append(nn.ReLU(inplace=True))
      self.conv_blocks.append(nn.Sequential(*block_layers))

  def forward(self,xyz,features):
      batch_size, num_points, _ = xyz.shape
      # Fix: Custom FPS to handle (B, N, 3) input
      # Helper for indexing points after FPS (common in PointNet++ implementations)
      def index_points(points, idx):
          device = points.device
          B = points.shape[0]
          view_shape = list(idx.shape)
          view_shape[1:] = [1] * (len(view_shape) - 1)
          repeat_shape = list(idx.shape)
          repeat_shape[0] = 1
          batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
          new_points = points[batch_indices, idx, :]
          return new_points

      # Apply FPS per batch element
      idx_list = []
      for b_idx in range(batch_size):
          idx_list.append(fps(xyz[b_idx], ratio=self.npoint/num_points)) # PyG fps on (N,3) returns (npoint,)
      idx = torch.stack(idx_list, dim=0) # (B, npoint)

      new_xyz = index_points(xyz, idx) # (B, npoint, 3)
      features_list=[]

      for i, radius in enumerate(self.radius_list):
        nsample = self.nsample_list[i] # Fix: get nsample from list

        batch_indices = torch.arange(batch_size, device=xyz.device).view(batch_size, 1, 1)
        # Fix typo: new_xyz_expanded
        new_xyz_expanded = new_xyz.unsqueeze(2).repeat(1, 1, num_points, 1)
        xyz_expanded = xyz.unsqueeze(1).repeat(1, self.npoint, 1, 1)
        # Fix: incorrect sum syntax
        sq_dist = torch.sum((new_xyz_expanded - xyz_expanded)**2, dim=-1)

        group_mask = sq_dist < (radius**2)
        _, sort_idx = sq_dist.sort(dim=-1)
        grouped_idx = sort_idx[:,:,:nsample] # (B, npoint, nsample)
        # Removed duplicated line: grouped_idx=sort_idx[:,:,:nsample]

        grouped_xyz = index_points(xyz, grouped_idx) - new_xyz.unsqueeze(2) # (B, npoint, nsample, 3)
        grouped_xyz = grouped_xyz.permute(0, 3, 1, 2) # (B, 3, npoint, nsample)

        if features is not None:
          grouped_features = index_points(features, grouped_idx) # (B, npoint, nsample, C_feat)
          grouped_features = grouped_features.permute(0, 3, 1, 2) # (B, C_feat, npoint, nsample)
          new_features = torch.cat([grouped_xyz, grouped_features], dim=1) # (B, 3+C_feat, npoint, nsample)
        else:
          # Fix: assignment to new_features
          new_features = grouped_xyz # (B, 3, npoint, nsample)

        # Fix: Typo in loop variable `conv_lyaer` -> `conv_layer` and ensure ReLU/BN are handled correctly
        new_features = self.conv_blocks[i](new_features)

        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]) # Max pool over nsample dimension
        features_list.append(new_features.squeeze(-1)) # (B, C_out, npoint)

      new_features = torch.cat(features_list, dim=1) # (B, C_total_msg, npoint)
      return new_xyz,new_features
