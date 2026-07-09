import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Project to queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attention_weights, v)

        # Concatenate heads and project back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)

        return output, attention_weights

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Layer Normalization before attention
        norm_x = self.norm1(x)
        # Self-attention with residual connection
        attn_output, _ = self.attention(norm_x)
        attn_output = self.dropout(attn_output)
        x = x + attn_output  # Residual connection

        # Layer Normalization before feed-forward
        norm_x = self.norm2(x)
        # Feed-forward network with residual connection
        ff_output = self.feed_forward(norm_x)
        ff_output = self.dropout(ff_output)
        x = x + ff_output  # Residual connection

        return x

class LinearRNNBlock(nn.Module):
  def __init__(self,embed_dim):
    super().__init__()
    self.embed_dim=embed_dim
    self.W_x=nn.Linear(embed_dim,embed_dim,bias=False)
    self.W_h=nn.Linear(embed_dim,embed_dim,bias=False)
    self.activation=nn.Tanh()
  def forward(self,x,h_prev=None):
    batch_size,seq_len,_=x.size()
    if h_prev is None:
      h_prev=torch.zeros(batch_size,self.embed_dim,device=x.device)
    outputs=[]
    for t in range(seq_len):
      current_x=x[:,t,:]
      h_curr=self.activation(self.W_x(current_x)+self.W_h(h_prev))
      outputs.append(h_curr)
      h_prev=h_curr
    outputs=torch.stack(outputs,dim=1)
    return outputs,h_prev