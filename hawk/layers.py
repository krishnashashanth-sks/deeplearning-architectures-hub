import torch
import torch.nn as nn
import math

class AssociativeScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, Bx):
        # A: (batch_size, seq_len, hidden_dim, hidden_dim)
        # Bx: (batch_size, seq_len, hidden_dim)

        batch_size, seq_len, hidden_dim = Bx.shape

        # Initialize output tensor to store the hidden states h_t
        output = torch.empty_like(Bx)

        # Initialize the first hidden state
        # As per instructions, h_0 is assumed to be zero or absorbed, so h_1 = Bx[0]
        output[:, 0, :] = Bx[:, 0, :]

        # Perform sequential associative scan
        for t in range(1, seq_len):
            # h_t = A_t @ h_{t-1} + Bx_t
            # A[:, t, :, :] is (batch_size, hidden_dim, hidden_dim)
            # output[:, t-1, :] is (batch_size, hidden_dim)
            # Bx[:, t, :] is (batch_size, hidden_dim)

            # Unsqueeze output[:, t-1, :] to (batch_size, hidden_dim, 1) for bmm
            # Then squeeze back to (batch_size, hidden_dim) after multiplication
            output[:, t, :] = torch.bmm(A[:, t, :, :], output[:, t-1, :].unsqueeze(-1)).squeeze(-1) + Bx[:, t, :]

        # Save tensors for the backward pass
        ctx.save_for_backward(A, Bx, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        A, Bx, output = ctx.saved_tensors
        batch_size, seq_len, hidden_dim = Bx.shape

        grad_A = torch.zeros_like(A)
        grad_Bx = torch.zeros_like(Bx)

        # Initialize propagated gradient from the next timestep
        # This will accumulate gradients coming from later timesteps
        propagated_grad_from_next_timestep = torch.zeros(batch_size, hidden_dim, device=grad_output.device, dtype=grad_output.dtype)

        # Iterate backward through the sequence
        for t in range(seq_len - 1, -1, -1):
            # Current total gradient w.r.t. h_t
            # It's the sum of the direct gradient from grad_output and the propagated gradient from h_{t+1}
            d_ht = grad_output[:, t, :] + propagated_grad_from_next_timestep

            # Gradient w.r.t. Bx_t
            # Since h_t = A_t @ h_{t-1} + Bx_t, d(h_t)/d(Bx_t) = I, so dL/dBx_t = dL/dh_t
            grad_Bx[:, t, :] = d_ht

            if t > 0: # Gradients for A_t and h_{t-1} only apply for t > 0
                # Gradient w.r.t. A_t
                # h_t = A_t @ h_{t-1} + Bx_t
                # d(h_t)/d(A_t) = h_{t-1}^T
                # dL/dA_t = dL/dh_t @ h_{t-1}^T
                # d_ht: (batch_size, hidden_dim)
                # output[:, t-1, :]: (batch_size, hidden_dim)
                # Resulting grad_A[:, t, :, :]: (batch_size, hidden_dim, hidden_dim)
                grad_A[:, t, :, :] = torch.bmm(d_ht.unsqueeze(-1), output[:, t-1, :].unsqueeze(1))

                # Propagate gradient to h_{t-1}
                # d(h_t)/d(h_{t-1}) = A_t
                # dL/dh_{t-1} = A_t^T @ dL/dh_t
                # A[:, t, :, :].transpose(-1, -2): (batch_size, hidden_dim, hidden_dim)
                # d_ht.unsqueeze(-1): (batch_size, hidden_dim, 1)
                propagated_grad_from_next_timestep = torch.bmm(A[:, t, :, :].transpose(-1, -2), d_ht.unsqueeze(-1)).squeeze(-1)
            else:
                # For t=0, there's no A_0 or h_{-1} to compute gradients for (assuming h_0=0)
                # The propagated_grad_from_next_timestep should be handled as if it's the last step for h_0
                propagated_grad_from_next_timestep = torch.zeros(batch_size, hidden_dim, device=grad_output.device, dtype=grad_output.dtype)

        return grad_A, grad_Bx

def associative_scan(A: torch.Tensor, Bx: torch.Tensor) -> torch.Tensor:
    """Helper function to perform the associative scan using the custom autograd function."""
    return AssociativeScanFunction.apply(A, Bx)

class RGLRULayer(nn.Module):
  def __init__(self,input_dim:int,hidden_dim:int):
    super().__init__()
    self.input_dim=input_dim
    self.hidden_dim=hidden_dim
    self.proj_bx=nn.Linear(input_dim,hidden_dim)
    self.proj_a=nn.Linear(input_dim,hidden_dim)
    self.output_proj=nn.Identity()
    print(f"RGLRULayer initialized with input_dim={input_dim}, hidden_dim={hidden_dim}")
  def forward(self,x):
    batch_size,seq_len,_=x.shape
    Bx=self.proj_bx(x)
    A_diag=torch.sigmoid(self.proj_a(x))
    A=torch.diag_embed(A_diag)
    A=A.to(Bx.dtype)
    h=associative_scan(A,Bx)
    return self.output_proj(h)
  
class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)