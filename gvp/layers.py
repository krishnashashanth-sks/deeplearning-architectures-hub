import torch
import torch.nn as nn

def _norm_no_nan(x,axis=-1,keepdims=False,eps=1e-8,l2=True):
  if l2:
    return torch.sqrt(torch.sum(x*x,axis=axis,keepdims=keepdims)+eps)
  else:
    return torch.sum(torch.abs(x),axis=axis,keepdims=keepdims)+eps
    
class GVP(nn.Module):
  def __init__(self,n_in,n_out,vector_gate=True,h_dim=None):
    super(GVP,self).__init__()
    self.n_s_in,self.n_v_in=n_in
    self.n_s_out,self.n_v_out=n_out
    self.vector_gate=vector_gate
    if h_dim is None:
      h_dim=max(self.n_s_in,self.n_s_out)
    self.s_mlp=nn.Sequential(
        nn.Linear(self.n_s_in+self.n_v_in,h_dim),
        nn.ReLU(True),
        nn.Linear(h_dim,self.n_s_out)
    )
    self.v_linear=nn.Linear(self.n_v_in,self.n_v_out,bias=False)
    if self.vector_gate:
      self.wh=nn.Linear(self.n_s_out,self.n_v_out)
      self.wv=nn.Linear(self.n_v_out,self.n_v_out,bias=False)
  def forward(self,x):
    s,v=x
    v_norm=_norm_no_nan(v).flatten(start_dim=-1)
    s_in=torch.cat([s,v_norm],dim=-1)
    s_out=self.s_mlp(s_in)

    # Corrected vector transformation: Transpose to make n_v_in the last dimension,
    # apply linear layer, then transpose back.
    v_out = self.v_linear(v.transpose(-1, -2)).transpose(-1, -2)

    if self.vector_gate:
      gate=torch.sigmoid(self.wh(s_out)).unsqueeze(-1)
      # Apply wv with the same transpose logic
      v_out = self.wv(v_out.transpose(-1, -2)).transpose(-1, -2) * gate
    return s_out,v_out
  
class GVPGNNLayer(nn.Module):
  def __init__(self,node_in_dims,node_out_dims,msg_dims,aggr_fn=torch.sum):
    super(GVPGNNLayer,self).__init__() # Added parentheses here
    self.node_in_s,self.node_in_v=node_in_dims
    self.node_out_s,self.node_out_v=node_out_dims
    self.msg_s,self.msg_v=msg_dims
    self.aggr_fn=aggr_fn
    self.node_transform=GVP(node_in_dims,node_out_dims)
    self.message_transform=GVP(node_in_dims,msg_dims)
    self.combine_gvp=GVP((self.node_out_s+self.msg_s,self.node_out_v+self.msg_v),
        node_out_dims
    )
  def forward(self,x,edge_index):
    s_nodes,v_nodes=x
    num_nodes=s_nodes.shape[0]
    s_transformed,v_transformed=self.node_transform((s_nodes,v_nodes))
    s_send,v_send=self.message_transform((s_nodes[edge_index[0]],v_nodes[edge_index[0]])) # Corrected v_nodes indexing
    s_aggregated=torch.zeros(num_nodes,self.msg_s,device=s_nodes.device)
    v_aggregated=torch.zeros(num_nodes,self.msg_v,3,device=v_nodes.device)
    s_aggregated.index_add_(0,edge_index[1],s_send)
    v_aggregated.index_add_(0,edge_index[1],v_send)
    s_combined=torch.cat([s_transformed,s_aggregated],dim=-1)
    v_combined=torch.cat([v_transformed,v_aggregated],dim=-2)
    s_out,v_out=self.combine_gvp((s_combined,v_combined))
    return s_out,v_out
  
def _softmax_by_node(src,index,num_nodes):
  att_weights=torch.zeros_like(src)
  unique_targets=torch.unique(index)
  for target_node in unique_targets:
    mask=(index==target_node)
    segment_scores=src[mask]
    if segment_scores.numel()>0:
      att_weights[mask]=torch.softmax(segment_scores,dim=-1)
  return att_weights

class GVPAttentionLayer(nn.Module):
  def __init__(self,node_in_dims,node_out_dims,head_dims,num_heads=1):
    super(GVPAttentionLayer,self).__init__()
    if num_heads>1:
      print("Warning: GVPAttentionLayer currently supported only num_heads=1 effectively.For multi-head ,you'd typically project to num_heads*head_dims and combine")
    self.node_in_s,self.node_in_v=node_in_dims
    self.node_out_s,self.node_out_v=node_out_dims
    self.head_s,self.head_v=head_dims
    self.query_gvp=GVP(node_in_dims,head_dims)
    self.key_gvp=GVP(node_in_dims,head_dims)
    self.value_gvp=GVP(node_in_dims,head_dims)
    self.attention_mlp=nn.Sequential(
        nn.Linear(2*self.head_s+2*self.head_v,self.head_s),
        nn.ReLU(),
        nn.Linear(self.head_s,1)
    )
    self.output_gvp=GVP(head_dims,node_out_dims)
  def forward(self,x,edge_index):
    s_nodes,v_nodes=x
    num_nodes=s_nodes.shape[0]
    q_s,q_v=self.query_gvp((s_nodes,v_nodes))
    k_s,k_v=self.key_gvp((s_nodes,v_nodes))
    val_s,val_v=self.value_gvp((s_nodes,v_nodes))
    source_nodes,target_nodes=edge_index[0],edge_index[1]
    q_s_target,q_v_target=q_s[target_nodes],q_v[target_nodes]
    k_s_source,k_v_source=k_s[source_nodes],k_v[source_nodes] # Fixed: Should be k_v[source_nodes]
    val_s_source,val_v_source=val_s[source_nodes],val_v[source_nodes]
    scalar_q_s=q_s_target
    scalar_k_s=k_s_source
    scalar_q_v_norm=_norm_no_nan(q_v_target).flatten(start_dim=-1)
    scalar_k_v_norm=_norm_no_nan(k_v_source).flatten(start_dim=-1)
    attention_input=torch.cat([scalar_q_s,scalar_k_s,scalar_q_v_norm,scalar_k_v_norm],dim=-1)
    attention_scores=self.attention_mlp(attention_input).squeeze(-1)
    attention_weights=_softmax_by_node(attention_scores,target_nodes,num_nodes)
    weighted_val_s=(attention_weights.unsqueeze(-1)*val_s_source)
    weighted_val_v=(attention_weights.unsqueeze(-1).unsqueeze(-1)*val_v_source)
    s_aggregated=torch.zeros(num_nodes,self.head_s,device=s_nodes.device)
    v_aggregated=torch.zeros(num_nodes,self.head_v,3,device=v_nodes.device) # Fixed: Added 3rd dimension for vectors
    s_aggregated.index_add_(0,target_nodes,weighted_val_s)
    v_aggregated.index_add_(0,target_nodes,weighted_val_v)
    s_out,v_out=self.output_gvp((s_aggregated,v_aggregated))
    return s_out,v_out