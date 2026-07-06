import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x) # Using GELU as a common activation function in Transformers
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MixtureOfExperts(nn.Module):
  def __init__(self,d_model,num_experts,top_k,d_ff_expert_factor=4,dropout=0.1):
    super().__init__()
    self.d_model=d_model
    self.num_experts=num_experts
    self.top_k=top_k
    self.d_ff_expert=d_model*d_ff_expert_factor

    # Gating Network (Router)
    self.gating_network=nn.Linear(d_model,num_experts)

    # Expert Networks
    # Assuming FeedForwardNetwork is accessible in this scope
    self.expert_networks=nn.ModuleList(
        [FeedForwardNetwork(d_model,self.d_ff_expert,dropout)for _ in range(num_experts)]
    )

    self.dropout=nn.Dropout(dropout)

  def forward(self,x):
    batch_size,seq_len,d_model=x.shape

    # Corrected line: removed 'self' from view arguments
    flat_x=x.view(-1,d_model)

    raw_scores=self.gating_network(flat_x)
    gate_weights=F.softmax(raw_scores,dim=-1)

    top_k_weights,top_k_indices=torch.topk(gate_weights,self.top_k,dim=-1)

    # Normalize top_k weights to sum to 1 for each token
    top_k_weights=top_k_weights/top_k_weights.sum(dim=-1,keepdim=True)
    top_k_weights=self.dropout(top_k_weights)

    output=torch.zeros_like(flat_x)

    # Expert processing and aggregation
    for i,expert in enumerate(self.expert_networks):
      # Create a boolean mask for tokens routed to the current expert
      is_routed_to_expert=(top_k_indices==i)

      # Get flat indices of tokens that are routed to this expert (at any of the top_k positions)
      routed_token_flat_indices=torch.nonzero(is_routed_to_expert.any(dim=-1),as_tuple=True)[0]

      if routed_token_flat_indices.numel()>0:
        tokens_for_expert=flat_x[routed_token_flat_indices]
        expert_output=expert(tokens_for_expert)

        # Get the specific weights for the current expert for the routed tokens
        # `torch.where(is_routed_to_expert[routed_token_flat_indices])[1]` finds the column index (0 to top_k-1)
        # where the current expert `i` was selected for each routed token.
        routed_weights=top_k_weights[routed_token_flat_indices,torch.where(is_routed_to_expert[routed_token_flat_indices])[1]]

        weighted_expert_output=expert_output*routed_weights.unsqueeze(-1)

        # Add weighted output to the total output tensor
        output.index_add_(0,routed_token_flat_indices,weighted_expert_output)

    # Calculate expert_load and expert_probability for load balancing loss
    # expert_load: count how many times each expert was selected (across all top_k choices for all tokens)
    expert_load = torch.zeros(self.num_experts, device=x.device)
    for i in range(self.num_experts):
        expert_load[i] = (top_k_indices == i).float().sum()

    # expert_probability: average probability assigned to each expert (across all tokens)
    expert_probability = gate_weights.mean(dim=0)

    return output.view(batch_size,seq_len,d_model), expert_load, expert_probability

class JambaLayer(nn.Module):
  def __init__(self,d_model,num_heads,d_ff,d_state,d_conv,expand,num_experts,top_k,dropout=0.1):
    super().__init__()
    self.d_model=d_model
    self.hybrid_block=HybridBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        d_conv=d_conv,
        expand=expand,
        d_state=d_state,
        dropout=dropout
        )
    self.moe_layer=MixtureOfExperts(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        dropout=dropout
    )
    self.norm_moe=nn.LayerNorm(d_model)
    self.dropout_moe=nn.Dropout(dropout)

  def forward(self,x,mask=None):
    hybrid_output=self.hybrid_block(x,mask)
    normed_hybrid_output=self.norm_moe(hybrid_output)

    # The MoE layer now returns output, expert_load, and expert_probability
    moe_output, expert_load, expert_probability = self.moe_layer(normed_hybrid_output)

    # Combine hybrid_output and moe_output with residual connection and dropout
    output = hybrid_output + self.dropout_moe(moe_output)

    return output, expert_load, expert_probability
