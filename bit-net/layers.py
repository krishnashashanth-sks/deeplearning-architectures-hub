import torch
import torch.nn as nn
import torch.autograd as autograd

class Binarize(autograd.Function):
    """
    A custom PyTorch autograd Function to binarize tensors with a Straight-Through Estimator (STE).
    During forward pass, it binarizes input to {-1, 1}.
    During backward pass, gradients are passed through unchanged (STE).
    """

    @staticmethod
    def forward(ctx, input):
        # Store the input for backward pass (to calculate gradients correctly for STE)
        # For STE, we often just need the shape/type, or sometimes the input itself
        # for more complex STE variants. Here, we don't strictly need it if we're
        # just passing through, but it's good practice for general autograd.Function.
        ctx.save_for_backward(input)

        # Binarize to -1 or 1
        # For input values exactly 0, we can define the behavior (e.g., 1 or -1)
        # Here, we follow common practice where sign(0) = 1.0
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: Gradients are passed through unchanged.
        # This means the gradient of the binarization operation is treated as 1.
        input, = ctx.saved_tensors

        # We need to ensure the gradient is only passed where the input was within
        # the valid range for the 'identity' part of STE. For simple sign(),
        # this often means passing the grad_output directly, as the 'identity'
        # gradient is 1. Some implementations clip gradients for inputs outside a range.

        # A common STE implementation for sign() operation is to pass gradients unchanged.
        # Optionally, one might clip the input to a certain range before multiplying
        # with grad_output if using an STE like 'clip_by_value' for inputs.
        # For pure sign(), this is often sufficient:
        return grad_output

binarize=Binarize.apply()

class BinarizedLinear(nn.Module):
    """
    A custom Binarized Linear layer that binarizes its weights during the forward pass.
    The full-precision weights are stored and updated during backpropagation.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(BinarizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store full-precision weights as a learnable parameter
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier uniform distribution
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        # Binarize weights during the forward pass
        binarized_weight = binarize(self.weight)

        # Perform linear operation with binarized weights
        output = torch.nn.functional.linear(input, binarized_weight, self.bias)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class QuantizeActivations(autograd.Function):
    """
    A custom PyTorch autograd Function to binarize activations to {-1, 1} with STE.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Binarize to -1 or 1. If 0, common practice is to make it 1.
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: Gradients are passed through unchanged.
        # This means the gradient of the binarization operation is treated as 1.
        input, = ctx.saved_tensors

        # Optionally, one might clip the gradients based on input value for robustness.
        # For simple sign() STE, passing grad_output directly is standard.
        return grad_output

quantize_activations=QuantizeActivations.apply()

class BinarizedSelfAttention(nn.Module):
  def __init__(self,dim,num_heads=8,qkv_bias=False,attn_drop=0.,proj_drop=0.):
    super().__init__()
    self.num_heads=num_heads
    head_dim=dim//num_heads
    self.scale=head_dim**-0.5
    self.qkv=BinarizedLinear(dim,dim*3,bias=qkv_bias)
    self.attn_drop=nn.Dropout(attn_drop)
    self.proj=BinarizedLinear(dim,dim)
    self.proj_drop=nn.Dropout(proj_drop)
  def forward(self,x):
    B,N,C=x.shape
    qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
    q,k,v=qkv[0],qkv[1],qkv[2]
    q=quantize_activations(q)
    k=quantize_activations(k)
    v=quantize_activations(v)
    attn=(q @ k.transpose(-2,-1))*self.scale
    attn=attn.softmax(dim=-1)
    attn=self.attn_drop(attn)
    x=(attn @ v).transpose(1,2).reshape(B,N,C)
    x=self.proj(x)
    x=self.proj_drop(x)
    return x

class BinarizeMlp(nn.Module):
  def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
    super().__init__()
    out_features=out_features or in_features
    hidden_features=hidden_features or in_features
    self.fc1=BinarizedLinear(in_features,hidden_features)
    self.act=act_layer()
    self.fc2=BinarizedLinear(hidden_features,out_features)
    self.drop=nn.Dropout(drop)
  def forward(self, x): 
    x=self.fc1(x)
    x=quantize_activations(x)
    x=self.act(x)
    x=self.drop(x) 
    x=self.fc2(x)
    x=quantize_activations(x)
    return self.drop(x)

class BinarizedTransformerBlock(nn.Module):
  def __init__(self,dim,num_heads,mlp_ratio=4.,qkv_bias=False,drop=0.,attn_drop=0.,
               drop_path=0.,act_layer=nn.GELU,norm_layer=nn.LayerNorm):
    super().__init__()
    self.norm1=norm_layer(dim)
    self.attn=BinarizedSelfAttention(dim,num_heads,qkv_bias,attn_drop,drop)
    self.drop_path=nn.Identity()
    self.norm2=norm_layer(dim)
    mlp_hidden_dim=int(dim*mlp_ratio)
    self.mlp=BinarizeMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
  def forward(self,x):
    x=x+self.drop_path(self.attn(quantize_activations(self.norm1(x))))
    x=x+self.drop_path(self.mlp(quantize_activations(self.norm2(x)))) 
    return x