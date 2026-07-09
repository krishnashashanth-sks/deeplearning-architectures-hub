import torch
import torch.nn as nn
from layers import DCRNNCell

class DCRNN(nn.Module):
  def __init__(self,input_size,hidden_size,num_nodes,history_length,predict_length,K=2):
    super().__init__()
    self.input_size=input_size
    self.hidden_size=hidden_size
    self.num_nodes=num_nodes
    self.history_length=history_length
    self.predict_length=predict_length
    self.K=K
    self.dcrnn_cell=DCRNNCell(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_nodes=self.num_nodes,
        K=self.K,
    )
    self.output_layer=nn.Linear(self.hidden_size,self.input_size)
  def forward(self,input_sequences,diffusion_matrix_f,diffusion_matrix_b):
    batch_size=input_sequences.size(0)
    H=torch.zeros(batch_size,self.num_nodes,self.hidden_size,device=input_sequences.device)
    for t in  range(self.history_length):
      X_t=input_sequences[:,t,:,:]
      H=self.dcrnn_cell(X_t,H,diffusion_matrix_f,diffusion_matrix_b)
    predictions=[]
    current_input_for_decoder=H
    for t in range(self.predict_length):
      predicted_features=self.output_layer(current_input_for_decoder)
      predictions.append(predicted_features)
      H=self.dcrnn_cell(predicted_features,H,diffusion_matrix_f,diffusion_matrix_b)
      current_input_for_decoder=H
    predictions=torch.stack(predictions,dim=1)
    return predictions