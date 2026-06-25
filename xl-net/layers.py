import torch
import torch.nn as nn
import math
import torch.functional as F

class RelativePositionalEncoding(nn.Module):
  def __init__(self,d_model,config):
    super(RelativePositionalEncoding,self).__init__()
    self.d_model=d_model
    self.max_seq_len=config['max_seq_len']
    self.mem_len=config['mem_len']
    self.max_len_relative=2*self.max_seq_len+self.mem_len
    self.pos_embeddings=nn.Parameter(torch.empty(self.max_len_relative,self.d_model))
    nn.init.normal_(self.pos_embeddings,mean=0.0,std=0.02)
    self.r_w_bias=nn.Parameter(torch.empty(self.d_model))
    self.r_r_bias=nn.Parameter(torch.empty(self.d_model))
    nn.init.normal_(self.r_w_bias,mean=0.0,std=0.02)
    nn.init.normal_(self.r_r_bias,mean=0.0,std=0.02)
  def forward(self,qlen,klen):
    rel_pos_indices=torch.arange(klen-1,-qlen,-1.0,device=self.pos_embeddings.device)
    rel_pos_indices=rel_pos_indices.long()+(klen-1)
    pos_emb=self.pos_embeddings[rel_pos_indices]
    return pos_emb,self.r_w_bias,self.r_r_bias

import torch.nn.functional as F
class TwoStreamSelfAttention(nn.Module):
  def __init__(self,config):
    super(TwoStreamSelfAttention,self).__init__()
    self.d_model=config['d_model']
    self.n_head=config['n_head']
    self.d_head=self.d_model//self.n_head
    self.dropout=config['dropout']
    self.dropatt=config['dropatt']
    self.mem_len=config['mem_len']
    self.q_net_h=nn.Linear(self.d_model,self.n_head*self.d_head,bias=False)
    self.k_net_h=nn.Linear(self.d_model,self.n_head*self.d_head,bias=False)
    self.v_net_h=nn.Linear(self.d_model,self.n_head*self.d_head,bias=False)
    self.q_net_g=nn.Linear(self.d_model,self.n_head*self.d_head,bias=False)
    self.o_net=nn.Linear(self.n_head*self.d_head,self.d_model,bias=False)
    self.dropout_layer=nn.Dropout(self.dropout)
    self.attn_dropout_layer=nn.Dropout(self.dropatt)
    self.r_w_bias=nn.Parameter(torch.empty(self.n_head,self.d_head))
    nn.init.normal_(self.r_w_bias,mean=0.0,std=0.02)
    self.r_r_bias=nn.Parameter(torch.empty(self.n_head,self.d_head))
    nn.init.normal_(self.r_r_bias,mean=0.0,std=0.02)
  @staticmethod
  def _rel_shift(x, klen, zero_triu=False):
    # x: (batch_size, n_head, qlen, klen_padded) which is the AR tensor
    # This function performs the relative shift operation as described in the Transformer-XL/XLNet paper.
    # It effectively pads the last dimension with zeros and then reshapes/slices to create a shifted version.

    original_shape = x.shape
    # Pad the last dimension (klen) with one zero on the left
    # (batch_size, n_head, qlen, klen_padded + 1)
    x_padded = F.pad(x, (1, 0))

    # Reshape to facilitate the shift.
    # This view operation essentially moves the added padding to the second-to-last dimension (now `klen_padded+1`).
    # The original `qlen` dimension becomes the last.
    # (batch_size, n_head, klen_padded + 1, qlen)
    x_reshaped = x_padded.view(original_shape[0], original_shape[1], original_shape[3] + 1, original_shape[2])

    # Slice off the first element from the second-to-last dimension (klen_padded + 1)
    # This performs the actual "shift".
    # Then, reshape back to the original shape (batch_size, n_head, qlen, klen_padded).
    # We use .contiguous() to ensure memory layout is correct before .view(*original_shape)
    x_shifted = x_reshaped[:, :, 1:].contiguous().view(*original_shape)

    # Crop the last dimension to the target klen
    if x_shifted.size(-1) > klen:
        x_shifted = x_shifted[..., :klen]
    elif x_shifted.size(-1) < klen:
        # If the shifted tensor is smaller than target klen, it indicates a deeper issue
        # or implies padding with zeros is needed. For now, we assume cropping is the primary fix.
        pass

    if zero_triu:
      # This part is typically for attention score masking (e.g., in AC),
      # not usually for the relative positional encoding term (AR).
      # The current attn_mask in _compute_attention should handle causal masking.
      # However, if zero_triu is set, apply a mask.
      # The original code's `x.size()` was wrong. It should be (qlen, klen).
      mask = torch.triu(x.new_ones(x.size(2), x.size(3)), diagonal=1).bool()
      x_shifted = x_shifted.masked_fill(mask, 0) # Apply mask by filling with zeros

    return x_shifted
  def _compute_attention(self,q,k,v,pos_emb,attn_mask,r_w_bias,r_r_bias):
    # Explicitly expand r_w_bias and r_r_bias for broadcasting safety
    r_w_bias_expanded = r_w_bias.unsqueeze(0).unsqueeze(2) # (1, n_head, 1, d_head)
    r_r_bias_expanded = r_r_bias.unsqueeze(0).unsqueeze(2) # (1, n_head, 1, d_head)

    AC=torch.einsum('bnqd,bnkd->bnqk',(q+r_w_bias_expanded).type_as(k),k.type_as(q))
    AR=torch.einsum('bnqd,knd->bnqk',(q+r_r_bias_expanded).type_as(pos_emb),pos_emb.type_as(q))
    AR=self._rel_shift(AR, k.size(2)) # Fix: Pass k.size(2) which is klen for the attention keys
    attn_score=AC+AR
    attn_score=attn_score/math.sqrt(self.d_head)
    if attn_mask is not None:
      # Changed `attn_mask==0` to `attn_mask` to directly use True for masked positions
      attn_score=attn_score.masked_fill(attn_mask,-torch.inf)
    attn_prob=F.softmax(attn_score,dim=-1)
    attn_prob=self.attn_dropout_layer(attn_prob)
    attn_vec=torch.einsum('bnqk,bnkd->bnqd',attn_prob,v)
    return attn_vec
  def forward(self,h,g,pos_emb,attn_mask,mem=None):
    qlen=h.size(1)
    mems=None
    if self.mem_len>0:
      new_mem=h if mem is None else torch.cat([mem,h],dim=1)
      mems=new_mem[:,-self.mem_len:].detach()
    if mem is not None and mem.size(1)>0:
      h_with_mem=torch.cat([mem,h],dim=1)
      g_with_mem=torch.cat([mem,g],dim=1)
    else:
      h_with_mem=h
      g_with_mem=g
    klen=h_with_mem.size(1)
    h_q=self.q_net_h(h).view(-1,qlen,self.n_head,self.d_head).permute(0,2,1,3)
    h_k=self.k_net_h(h_with_mem).view(-1,klen,self.n_head,self.d_head).permute(0,2,1,3)
    h_v=self.v_net_h(h_with_mem).view(-1,klen,self.n_head,self.d_head).permute(0,2,1,3)
    g_q=self.q_net_g(g).view(-1,qlen,self.n_head,self.d_head).permute(0,2,1,3)
    pos_emb_for_attn=pos_emb.view(pos_emb.size(0),self.n_head,self.d_head)
    attn_vec_h=self._compute_attention(
        q=h_q,k=h_k,v=h_v,
        pos_emb=pos_emb_for_attn,
        attn_mask=attn_mask,
        r_w_bias=self.r_w_bias,
        r_r_bias=self.r_r_bias,
    )
    attn_vec_h=attn_vec_h.permute(0,2,1,3).contiguous().view(-1,qlen,self.n_head*self.d_head)
    attn_vec_g=self._compute_attention(
        q=g_q,k=h_k,v=h_v,
        pos_emb=pos_emb_for_attn,
        attn_mask=attn_mask,
        r_w_bias=self.r_w_bias,
        r_r_bias=self.r_r_bias,
    )
    attn_vec_g=attn_vec_g.permute(0,2,1,3).contiguous().view(-1,qlen,self.n_head*self.d_head)
    h_new=self.o_net(attn_vec_g)
    h_new=self.dropout_layer(h_new)
    g_new=self.o_net(attn_vec_g)
    g_new=self.dropout_layer(g_new)
    return h_new,g_new,mems
        
class TransformerXLBlock(nn.Module):
  def __init__(self,config):
    super(TransformerXLBlock,self).__init__()
    self.attn=TwoStreamSelfAttention(config)
    self.d_model=config['d_model']
    self.d_inner=config['d_inner']
    self.dropout=config['dropout']
    self.ffn=nn.Sequential(
        nn.Linear(self.d_model,self.d_inner),
        nn.GELU(),
        nn.Dropout(self.dropout),
        nn.Linear(self.d_inner,self.d_model),
        nn.GELU(),
        nn.Dropout(self.dropout)
    )
    self.layer_norm_attn_h=nn.LayerNorm(self.d_model)
    self.layer_norm_attn_g=nn.LayerNorm(self.d_model)
    self.layer_norm_ffn_h=nn.LayerNorm(self.d_model)
    self.layer_norm_ffn_g=nn.LayerNorm(self.d_model)
    self.dropout_attn_h=nn.Dropout(self.dropout)
    self.dropout_attn_g=nn.Dropout(self.dropout)
    self.dropout_ffn_h=nn.Dropout(self.dropout)
    self.dropout_ffn_g=nn.Dropout(self.dropout)
  def forward(self,h,g,pos_emb,attn_mask,mems=None):
    norm_h=self.layer_norm_attn_h(h)
    norm_g=self.layer_norm_attn_g(g)
    h_attn,g_attn,new_mems=self.attn(norm_h,norm_g,pos_emb,attn_mask,mem=mems)
    h_attn_res=h+self.dropout_attn_h(h_attn)
    g_attn_res=g+self.dropout_attn_g(g_attn)
    norm_h_ffn=self.layer_norm_ffn_h(h_attn_res)
    norm_g_ffn=self.layer_norm_ffn_g(g_attn_res)
    h_ffn=self.ffn(norm_h_ffn)
    g_ffn=self.ffn(norm_g_ffn)
    h_out=h_attn_res+self.dropout_ffn_h(h_ffn)
    g_out=g_attn_res+self.dropout_ffn_g(g_ffn)
    return  h_out,g_out,new_mems