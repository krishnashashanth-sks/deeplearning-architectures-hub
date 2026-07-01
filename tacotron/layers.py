import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
  def __init__(self,in_dim,out_dim,bias=True):
    super().__init__()
    self.linear_layer=nn.Linear(in_dim,out_dim,bias=bias)
  def forward(self,x):
    return self.linear_layer(x)

class Conv1d(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=None,dilation=1,bias=False):
    super(Conv1d,self).__init__()
    if padding is None:
      padding=int((kernel_size*dilation-dilation)/2)
    self.conv=nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias)
  def forward(self,x):
    return self.conv(x)
  
class Encoder(nn.Module):
  def __init__(self,hparams):
    super(Encoder,self).__init__()
    self.embedding=nn.Embedding(hparams.n_symbols,hparams.symbols_embedding_dim)
    convolutions=[]
    for i in range(hparams.encoder_n_convolution):
      conv_layer=nn.Sequential(
          Conv1d(hparams.symbols_embedding_dim if i==0 else hparams.encoder_kernel_size,hparams.encoder_kernel_size,kernel_size=hparams.encoder_kernel_size,padding=int((hparams.encoder_kernel_size-1)/2)),
          nn.BatchNorm1d(hparams.encoder_kernel_size)
      )
      convolutions.append(conv_layer)
    self.convolutions=nn.ModuleList(convolutions)
    self.lstm=nn.LSTM(hparams.encoder_kernel_size,int(hparams.encoder_hidden_dim/2),1,bidirectional=True)
  def forward(self,text_input):
    embedded_inputs=self.embedding(text_input).transpose(1,2)
    for conv in self.convolutions:
      embedded_inputs=F.dropout(F.relu(conv(embedded_inputs)),0.5,self.training)
    embedded_inputs=embedded_inputs.transpose(1,2)
    outputs,_=self.lstm(embedded_inputs)
    return outputs

class Attention(nn.Module):
  def __init__(self,hparams):
    super(Attention,self).__init__()
    self.query_layer=Linear(hparams.decoder_rnn_dim,hparams.attention_rnn_dim,bias=False)
    self.memory_layer=Linear(hparams.encoder_hidden_dim,hparams.attention_rnn_dim,bias=False) # Corrected: hparams.decoder_rnn_dim -> hparams.encoder_hidden_dim
    self.v=Linear(hparams.attention_rnn_dim,1,bias=False)
    self.location_conv=Conv1d(1,hparams.attention_location_n_filters,kernel_size=hparams.attention_location_kernel_size,bias=False,stride=1,padding=int((hparams.attention_location_kernel_size-1)/2))
    self.location_dense=Linear(hparams.attention_location_n_filters,hparams.attention_rnn_dim,bias=False)
    self.score_mask_value=-float('inf')
  def get_alignment_energies(self,query,processed_memory,processed_location_features,mask):
    processed_query=self.query_layer(query)
    energies=self.v(torch.tanh(processed_query+processed_memory+processed_location_features))
    energies=energies.squeeze(-1)
    if mask is not None:
      energies.data.masked_fill_(mask,self.score_mask_value)
    return energies
  def forward(self,query,memory,processed_memory,attention_weights_cat,mask):
    processed_location_features=self.location_dense(self.location_conv(attention_weights_cat.unsqueeze(1)).transpose(1,2))
    energies=self.get_alignment_energies(query,processed_memory,processed_location_features,mask)
    attention_weights=F.softmax(energies,dim=1)
    context=torch.bmm(attention_weights.unsqueeze(1),memory)
    context=context.squeeze(1)
    return context,attention_weights

class Prenet(nn.Module):
  def __init__(self,in_dim,sizes):
    super(Prenet,self).__init__()
    in_sizes=[in_dim]+sizes[:-1]
    self.layers=nn.ModuleList(
        [Linear(in_size,out_size) for (in_size,out_size) in zip(in_sizes,sizes)]
    )
  def forward(self,x):
    for linear in self.layers:
      x=F.dropout(F.relu(linear(x)),p=0.5,training=self.training)
    return x

class Postnet(nn.Module):
  def __init__(self,hparams):
    super(Postnet,self).__init__()
    self.convolutions=nn.ModuleList()

    # First layer: input_dim = n_mel_channels, output_dim = postnet_embedding_dim
    self.convolutions.append(
        nn.Sequential(
            Conv1d(hparams.n_mel_channels,
                   hparams.postnet_embedding_dim,
                   kernel_size=hparams.postnet_kernel_size,
                   padding=int((hparams.postnet_kernel_size-1)/2),bias=False),
            nn.BatchNorm1d(hparams.postnet_embedding_dim),
            nn.Tanh()
        )
    )

    # Intermediate layers: input_dim = postnet_embedding_dim, output_dim = postnet_embedding_dim
    for i in range(1, hparams.postnet_n_convolution - 1):
      self.convolutions.append(
          nn.Sequential(
              Conv1d(hparams.postnet_embedding_dim,
                     hparams.postnet_embedding_dim,
                     kernel_size=hparams.postnet_kernel_size,
                     padding=int((hparams.postnet_kernel_size-1)/2),bias=False),
              nn.BatchNorm1d(hparams.postnet_embedding_dim),
              nn.Tanh()
          )
      )

    # Last layer: input_dim = postnet_embedding_dim, output_dim = n_mel_channels
    self.convolutions.append(
        nn.Sequential(
            Conv1d(hparams.postnet_embedding_dim,
                   hparams.n_mel_channels,
                   kernel_size=hparams.postnet_kernel_size,
                   padding=int((hparams.postnet_kernel_size-1)/2),bias=False),
            nn.BatchNorm1d(hparams.n_mel_channels)
        )
    )

  def forward(self,x):
    for i in range(len(self.convolutions)-1):
      x=F.dropout(self.convolutions[i](x),0.5,self.training)
    x=F.dropout(self.convolutions[-1](x),0.5,self.training)
    return x
  
class Decoder(nn.Module):
  def __init__(self,hparams):
    super(Decoder,self).__init__()
    self.n_mel_channels=hparams.n_mel_channels
    self.n_frames_per_step=hparams.n_frames_per_step
    self.encoder_hidden_dim=hparams.encoder_hidden_dim
    self.decoder_rnn_dim=hparams.decoder_rnn_dim # Corrected typo: decoder_rnn_sim -> decoder_rnn_dim
    self.prenet_dims=hparams.prenet_dims
    self.max_decoder_steps=hparams.max_decoder_steps
    self.gate_threshold=hparams.gate_threshold
    self.prenet=Prenet(hparams.n_mel_channels*hparams.n_frames_per_step,hparams.prenet_dims) # Corrected typo: prent -> prenet, n_nel_channels -> n_mel_channels, prenet_dim -> prenet_dims
    self.attention_lstm=nn.LSTMCell(hparams.prenet_dims[-1]+hparams.encoder_hidden_dim,hparams.decoder_rnn_dim)
    self.attention_layer=Attention(hparams)
    self.decoder_lstm=nn.LSTMCell(hparams.decoder_rnn_dim+hparams.encoder_hidden_dim,hparams.decoder_rnn_dim)
    self.linear_projection=Linear(hparams.decoder_rnn_dim+hparams.encoder_hidden_dim,hparams.n_mel_channels*hparams.n_frames_per_step) # Corrected typo: linear_projeciton -> linear_projection
    self.gate_layer=Linear(hparams.decoder_rnn_dim+hparams.encoder_hidden_dim,1,bias=True)

  def decode(self, decoder_input,attention_hidden,attention_cell,decoder_hidden,decoder_cell,attention_weights,attention_context,memory,processed_memory,mask): # Corrected signature: self.decoder_input -> decoder_input, processed_memmory -> processed_memory
    cell_input=torch.cat([decoder_input,attention_context],-1)
    attention_hidden,attention_cell=self.attention_lstm(cell_input,(attention_hidden,attention_cell))
    attention_hidden=F.dropout(attention_hidden,0.1,self.training) # Corrected typo: dropotu -> dropout
    attention_context,attention_weights=self.attention_layer(
        attention_hidden.unsqueeze(1),memory,processed_memory,attention_weights,mask
    )
    attention_context=attention_context.squeeze(1)
    decoder_input_from_attention=torch.cat((attention_hidden,attention_context),-1)
    decoder_hidden,decoder_cell=self.decoder_lstm(decoder_input_from_attention,(decoder_hidden,decoder_cell))
    decoder_hidden=F.dropout(decoder_hidden,0.1,self.training)
    decoder_output=torch.cat((decoder_hidden,attention_context),-1)
    mel_output=self.linear_projection(decoder_output)
    gate_output=self.gate_layer(decoder_output)
    return mel_output,gate_output,attention_hidden,attention_cell,decoder_hidden,decoder_cell,attention_weights,attention_context

  def forward(self,memory,decoder_inputs,memory_lengths):
    batch_size=memory.size(0)
    go_frame=memory.new_zeros(batch_size,self.n_mel_channels*self.n_frames_per_step).unsqueeze(0)
    decoder_inputs=torch.cat((go_frame,decoder_inputs),dim=0)
    decoder_inputs=self.prenet(decoder_inputs.view(-1,self.n_mel_channels*self.n_frames_per_step))
    decoder_inputs=decoder_inputs.view(-1,batch_size,self.prenet_dims[-1])
    attention_hidden=memory.new_zeros(batch_size,self.decoder_rnn_dim)
    attention_cell=memory.new_zeros(batch_size,self.decoder_rnn_dim)
    decoder_hidden=memory.new_zeros(batch_size,self.decoder_rnn_dim)
    decoder_cell=memory.new_zeros(batch_size,self.decoder_rnn_dim) # Corrected typo: batch-size -> batch_size
    attention_weights=memory.new_zeros(batch_size,memory.size(1)) # Corrected typo: attnetion_weights -> attention_weights
    attention_context=memory.new_zeros(batch_size,self.encoder_hidden_dim)
    processed_memory=self.attention_layer.memory_layer(memory) # Corrected typo: processed_memmory -> processed_memory
    mask=self.get_mask_from_lengths(memory_lengths, memory.size(1))
    mel_outputs,gate_outputs,alignments=[],[],[] # Corrected typo: mel_outpus -> mel_outputs
    while len(mel_outputs)<decoder_inputs.size(0)-1:
      decoder_input=decoder_inputs[len(mel_outputs)]
      mel_output,gate_output,attention_hidden,attention_cell, \
      decoder_hidden,decoder_cell,attention_weights,attention_context = \
      self.decode(decoder_input,attention_hidden,attention_cell,decoder_hidden,decoder_cell,
                   attention_weights,attention_context,memory,processed_memory,mask)
      mel_outputs+=[mel_output.squeeze(1)]
      gate_outputs+=[gate_output.squeeze(1)]
      alignments+=[attention_weights]
    mel_outputs=torch.stack(mel_outputs)
    gate_outputs=torch.stack(gate_outputs) # Corrected typo: gate_otuputs -> gate_outputs
    alignments=torch.stack(alignments)
    mel_outputs=mel_outputs.transpose(0,1).contiguous().view( # Corrected typo: contingous() -> contiguous()
        batch_size,self.n_mel_channels,-1)
    return mel_outputs,gate_outputs,alignments # Corrected typo: returnmel_output -> return mel_outputs

  def inference(self,memory,memory_lengths):
    batch_size=memory.size(0)
    go_frame=memory.new_zeros(batch_size,self.n_mel_channels*self.n_frames_per_step)
    attention_hidden=memory.new_zeros(batch_size,self.decoder_rnn_dim) # Corrected typo: batch-size -> batch_size
    attention_cell=memory.new_zeros(batch_size,self.decoder_rnn_dim)
    decoder_hidden=memory.new_zeros(batch_size,self.decoder_rnn_dim)
    decoder_cell=memory.new_zeros(batch_size,self.decoder_rnn_dim)
    attention_context=memory.new_zeros(batch_size,self.encoder_hidden_dim)
    attention_weights=memory.new_zeros(batch_size,memory.size(1))
    processed_memory=self.attention_layer.memory_layer(memory)
    mask=self.get_mask_from_lengths(memory_lengths, memory.size(1))
    mel_outputs,gate_outputs,alignments=[],[],[] # Corrected typo: mel_outptus -> mel_outputs
    decoder_input=go_frame
    while True:
      decoder_input=self.prenet(decoder_input)
      mel_output,gate_output,attention_hidden,attention_cell,decoder_hidden,decoder_cell,attention_weights,attention_context = \
      self.decode(decoder_input,attention_hidden,attention_cell,decoder_hidden,decoder_cell,attention_weights,attention_context,memory,processed_memory,mask) # Corrected typo: attentions_weights, attentions_context -> attention_weights, attention_context
      mel_outputs+=[mel_output.squeeze(1)] # Corrected typo: mel_outptus -> mel_outputs
      gate_outputs+=[gate_output]
      alignments+=[attention_weights]
      if(torch.sigmoid(gate_output)>self.gate_threshold).all() or len(mel_outputs)==self.max_decoder_steps:
        break
      decoder_input=mel_output
    mel_outputs=torch.stack(mel_outputs).transpose(0,1).contiguous().view(batch_size,self.n_mel_channels,-1) # Corrected typo: mel_outptus -> mel_outputs, batch-size -> batch_size
    gate_outputs=torch.stack(gate_outputs).transpose(0,1)
    alignments=torch.stack(alignments).transpose(0,1)
    return mel_outputs,gate_outputs,alignments # Corrected typo: mel_outptus -> mel_outputs

  def get_mask_from_lengths(self, lengths, max_len):
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return ~mask