from layers import *
import torch.nn as nn
import torch

class XLNetModel(nn.Module):
  def __init__(self,config):
    super(XLNetModel,self).__init__()
    self.vocab_size=config['vocab_size']
    self.d_model=config['d_model']
    self.n_head=config['n_head']
    self.d_head=self.d_model//self.n_head
    self.n_layer=config['n_layer']
    self.mem_len=config['mem_len']
    self.max_seq_len=config['max_seq_len']
    self.dropout=config['dropout']
    self.word_embedding=nn.Embedding(self.vocab_size,self.d_model)
    self.pos_emb=RelativePositionalEncoding(self.d_model,config)
    self.layers=nn.ModuleList(
        TransformerXLBlock(config) for _ in range(self.n_layer)
    )
    self.dropout_in=nn.Dropout(self.dropout)
    self.dropout_out=nn.Dropout(self.dropout)
  def forward(self,input_ids,attention_mask=None,mems=None,perm_mask=None):
    qlen=input_ids.size(1)
    # Fix: Ensure mems[0] is not None before calling .size(1)
    mlen = mems[0].size(1) if mems is not None and len(mems) > 0 and mems[0] is not None else 0
    klen=qlen+mlen
    word_emb=self.word_embedding(input_ids)
    h=self.dropout_in(word_emb)
    g=self.dropout_in(torch.zeros_like(word_emb)) # Query stream init
    pos_emb,r_w_bias,r_r_bias=self.pos_emb(qlen,klen)

    # 3. Create attention mask (True means masked, False means valid)
    # Causal mask: True for positions to be masked (future tokens), False for valid (past/current tokens)
    causal_mask_local = torch.triu(h.new_ones(qlen, qlen), diagonal=1).bool() # (qlen, qlen)

    # Extend causal mask to include memory. Memory part should be False (valid).
    if mlen > 0:
      memory_mask_local = h.new_zeros(qlen, mlen).bool() # (qlen, mlen) with False (valid)
      causal_mask_local = torch.cat([memory_mask_local, causal_mask_local], dim=1) # (qlen, klen) pattern

    # Initialize `final_attn_mask` for all batches by broadcasting `causal_mask_local`. True means masked.
    final_attn_mask = causal_mask_local.unsqueeze(0).repeat(h.size(0), 1, 1) # (batch_size, qlen, klen)

    if attention_mask is not None: # User-provided attention mask (padding mask)
      # attention_mask: (batch_size, qlen), 1 for valid token, 0 for padding.
      # We need a mask that is True for query positions that are padding (to be masked).
      input_padding_mask_for_queries = (attention_mask == 0).unsqueeze(-1).expand(-1, -1, klen).bool() # (batch_size, qlen, klen)
      final_attn_mask = final_attn_mask | input_padding_mask_for_queries # Logical OR: True if either source masks

    if perm_mask is not None: # Permutation mask for PLM
      # perm_mask: (batch_size, qlen, qlen), True means mask for this permutation step.
      if mlen > 0:
        # Extend perm_mask for memory. Memory part should be False (valid for permutation attention).
        perm_mask_ext = torch.cat([perm_mask.new_zeros(perm_mask.size(0), qlen, mlen).bool(), perm_mask], dim=2) # (batch_size, qlen, klen)
      else:
        perm_mask_ext = perm_mask # (batch_size, qlen, klen)
      final_attn_mask = final_attn_mask | perm_mask_ext # Logical OR

    # Final attention mask to pass to layers: (batch_size, 1, qlen, klen). True means masked.
    attn_mask_to_pass = final_attn_mask.unsqueeze(1) # (batch_size, 1, qlen, klen)

    new_mems=[]
    for i,layer in enumerate(self.layers):
      current_mem=mems[i] if mems is not None and i<len(mems)and mems[i] is not None else None # Fix: Also check mems[i] for None
      h,g,block_new_mems=layer(h,g,pos_emb,attn_mask_to_pass,current_mem) # Pass the new mask variable
      new_mems.append(block_new_mems)

    h=self.dropout_out(h)
    g=self.dropout_out(g)
    return h,g,new_mems

class PermutationLanguageModel(nn.Module):
  def __init__(self,config):
    super(PermutationLanguageModel,self).__init__()
    self.d_model=config['d_model']
    self.vocab_size=config['vocab_size']
    self.proj=nn.Linear(self.d_model,self.vocab_size)
    self.loss_fn=nn.CrossEntropyLoss(ignore_index=-1)
  def forward(self,g,target_ids,perm_mask):
    if perm_mask.dtype!=torch.bool:
      perm_mask=perm_mask.bool()
    g_masked=g[perm_mask]
    logits=self.proj(g_masked)
    target=target_ids[perm_mask]
    loss=self.loss_fn(logits,target)
    return loss,logits