import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

def monarch_matrix_operation(W_dense, num_blocks, intermediate_dim_per_block):
    """
    Implements a simplified Monarch Matrix decomposition and reconstruction.
    This function decomposes a given dense matrix W into two block-diagonal matrices
    A and B, and then reconstructs an approximation of W by multiplying A and B.

    The decomposition involves:
    1. Splitting the input matrix `W_dense` into `num_blocks` diagonal sub-matrices.
    2. For each sub-matrix, performing Singular Value Decomposition (SVD) to obtain
       a low-rank approximation, thereby generating the corresponding blocks for A and B.
    3. Constructing the full block-diagonal matrices A and B from these blocks.

    The reconstructed matrix `W_reconstructed` will be block-diagonal, approximating
    `W_dense` by its block-diagonal components.

    Args:
        W_dense (torch.tensor): The large input matrix to decompose. Shape (N, M).
        num_blocks (int): The number of blocks for the Monarch decomposition.
                          N and M must be divisible by num_blocks.
        intermediate_dim_per_block (int): The intermediate dimension for each
                                          block in the decomposition (k_i in M_i^1 @ M_i^2).
                                          Must be less than or equal to the minimum of
                                          `block_h` and `block_w` (dimensions of each diagonal block).

    Returns:
        tuple: A tuple containing:
            - A (torch.tensor): The first block-diagonal matrix (e.g., M2 in W = M2 @ M1).
            - B (torch.tensor): The second block-diagonal matrix (e.g., M1 in W = M2 @ M1).
            - W_reconstructed (torch.tensor): The reconstructed matrix (A @ B).
    """
    N, M = W_dense.shape

    if N % num_blocks != 0 or M % num_blocks != 0:
        raise ValueError("Dimensions of W_dense must be divisible by num_blocks.")

    block_h = N // num_blocks
    block_w = M // num_blocks

    if intermediate_dim_per_block > min(block_h, block_w):
        raise ValueError("intermediate_dim_per_block must be less than or equal to the minimum of block_h and block_w for a meaningful SVD decomposition.")

    A_blocks = []
    B_blocks = []

    for i in range(num_blocks):
        # Extract the i-th diagonal block from W_dense
        W_sub = W_dense[i * block_h : (i + 1) * block_h, i * block_w : (i + 1) * block_w]

        # Perform Singular Value Decomposition (SVD) for low-rank approximation
        # This is a common mathematical decomposition technique to find A_i and B_i such that W_sub ~ A_i @ B_i
        U, S, Vh = torch.linalg.svd(W_sub, full_matrices=False)

        # Truncate SVD to intermediate_dim_per_block to get the low-rank approximation
        k = intermediate_dim_per_block
        U_k = U[:, :k]
        S_k_sqrt = torch.diag_embed(torch.sqrt(S[:k]))
        Vh_k = Vh[:k, :]

        # Define A_i and B_i blocks. A_i will have shape (block_h, k), B_i will have shape (k, block_w)
        # This split ensures A_i @ B_i approximates W_sub.
        A_i = U_k @ S_k_sqrt
        B_i = S_k_sqrt @ Vh_k

        A_blocks.append(A_i)
        B_blocks.append(B_i)

    # Construct the full block-diagonal matrices A and B using torch.block_diag
    # A will have total shape (N, num_blocks * intermediate_dim_per_block)
    # B will have total shape (num_blocks * intermediate_dim_per_block, M)
    A = torch.block_diag(*A_blocks)
    B = torch.block_diag(*B_blocks)

    # Reconstruct the matrix W_reconstructed by multiplying A and B
    # W_reconstructed will be block-diagonal, approximating the diagonal blocks of W_dense
    W_reconstructed = A @ B

    return A, B, W_reconstructed

class MonarchConv(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,num_blocks,intermediate_dim_per_block):
    super(MonarchConv,self).__init__()
    self.in_channels=in_channels
    self.out_channels=out_channels
    self.kernel_size=kernel_size
    self.num_blocks=num_blocks
    self.intermediate_dim_per_block=intermediate_dim_per_block
    N_flat=out_channels
    M_flat=in_channels*kernel_size
    if N_flat % num_blocks !=0 or M_flat % num_blocks !=0:
       raise ValueError(
                f"Dimensions N_flat ({N_flat}) and M_flat ({M_flat}) "
                f"must be divisible by num_blocks ({num_blocks})."
            )
    block_h=N_flat//num_blocks
    block_w=M_flat//num_blocks
    if intermediate_dim_per_block>min(block_h,block_w):
      raise ValueError(
                f"intermediate_dim_per_block ({intermediate_dim_per_block}) must be "
                f"less than or equal to min(block_h={block_h}, block_w={block_w})."
            )
    self.A_blocks=nn.Parameter(torch.empty(num_blocks,block_h,intermediate_dim_per_block))
    self.B_blocks=nn.Parameter(torch.empty(num_blocks,intermediate_dim_per_block,block_w))
    nn.init.kaiming_uniform_(self.A_blocks) # Removed 'a' parameter
    nn.init.kaiming_uniform_(self.B_blocks) # Removed 'a' parameter
  def _build_monarch_kernel(self):
    A_list=[self.A_blocks[i] for i in range(self.num_blocks)]
    B_list=[self.B_blocks[i] for i in range(self.num_blocks)]
    A=torch.block_diag(*A_list)
    B=torch.block_diag(*B_list)
    W_reconstructed_flat=A @ B
    W_reconstructed=W_reconstructed_flat.view(
        self.out_channels,self.in_channels,self.kernel_size
    )
    return W_reconstructed
  def forward(self,x):
    monarch_kernel=self._build_monarch_kernel()
    return F.conv1d(x,monarch_kernel,stride=1,padding=self.kernel_size//2)
  
class MonarchMLP(nn.Module):
  def __init__(self,in_features,out_features,num_blocks,intermediate_dim_per_block,bias=True):
    super(MonarchMLP,self).__init__()
    self.in_features=in_features
    self.out_features=out_features
    self.num_blocks=num_blocks
    self.intermediate_dim_per_block=intermediate_dim_per_block

    N_flat=out_features
    M_flat=in_features

    if N_flat % num_blocks!=0 or M_flat% num_blocks!=0:
      raise ValueError(
                f"Dimensions N_flat ({N_flat}) and M_flat ({M_flat}) "
                f"must be divisible by num_blocks ({num_blocks})."
            )

    block_h=N_flat//num_blocks
    block_w=M_flat//num_blocks

    if intermediate_dim_per_block>min(block_h,block_w):
      raise ValueError(
                f"intermediate_dim_per_block ({intermediate_dim_per_block}) must be "
                f"less than or equal to min(block_h={block_h}, block_w={block_w})."
      )

    self.A_blocks=nn.Parameter(torch.empty(num_blocks,block_h,intermediate_dim_per_block))
    self.B_blocks=nn.Parameter(torch.empty(num_blocks,intermediate_dim_per_block,block_w))

    if bias:
      self.bias=nn.Parameter(torch.empty(out_features))
    else:
      self.register_parameter('bias',None)

    self.reset_parameters()

  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.A_blocks)
    nn.init.kaiming_uniform_(self.B_blocks)
    if self.bias is not None:
      fan_in=self.in_features
      bound=1/torch.sqrt(torch.tensor(fan_in)) if fan_in > 0 else 0
      nn.init.uniform_(self.bias,-bound,bound)

  def _build_monarch_weight(self):
    A_list=[self.A_blocks[i] for i in range(self.num_blocks)]
    B_list=[self.B_blocks[i] for i in range(self.num_blocks)]
    A=torch.block_diag(*A_list)
    B=torch.block_diag(*B_list)
    return A @ B

  def forward(self,x):
    monarch_weight=self._build_monarch_weight()
    output=F.linear(x,monarch_weight,self.bias)
    return output
  
class M2Block(nn.Module):
  def __init__(self,embed_dim,kernel_size,
               num_blocks_conv,intermediate_dim_conv,
               num_blocks_mlp,intermediate_dim_mlp, # Corrected parameter name
               dropout_rate):
    super(M2Block,self).__init__()
    self.embed_dim=embed_dim
    self.monarch_conv=MonarchConv(
        in_channels=embed_dim,
        out_channels=embed_dim,
        kernel_size=kernel_size,
        num_blocks=num_blocks_conv,
        intermediate_dim_per_block=intermediate_dim_conv
    )
    self.norm1=nn.LayerNorm(embed_dim)
    self.activation=nn.GELU()
    self.monarch_mlp=MonarchMLP(
        in_features=embed_dim,
        out_features=embed_dim,
        num_blocks=num_blocks_mlp, # Corrected parameter name
        intermediate_dim_per_block=intermediate_dim_mlp
    )
    self.norm2=nn.LayerNorm(embed_dim)
    self.dropout=nn.Dropout(dropout_rate)
  def forward(self,x):
    identity=x
    conv_out=self.monarch_conv(x)
    normed1_out=self.norm1(conv_out.permute(0,2,1)).permute(0,2,1)
    activated_out=self.activation(normed1_out)
    batch_size,_,seq_len=activated_out.shape
    mlp_input=activated_out.permute(0,2,1).reshape(-1,self.embed_dim) # Corrected reshape
    mlp_out=self.monarch_mlp(mlp_input)
    mlp_out=mlp_out.reshape(batch_size,seq_len,self.embed_dim).permute(0,2,1)
    normed2_out=self.norm2(mlp_out.permute(0,2,1)).permute(0,2,1)
    dropout_out=self.dropout(normed2_out)
    return dropout_out+identity