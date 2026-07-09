import torch
import torch.nn as nn
import torch.nn.functional as F

class TextFeatureExtractor(nn.Module):
  def __init__(self, vocab_size, embed_dim, text_hidden_dim, num_filters, kernel_sizes):
    super(TextFeatureExtractor, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.conv_layers = nn.ModuleList([
        nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
    ])
    self.fc = nn.Linear(len(kernel_sizes) * num_filters, text_hidden_dim)

  def forward(self, x):
    # x shape: (batch_size, sequence_length)
    embedded = self.embedding(x)
    # embedded shape: (batch_size, sequence_length, embed_dim)
    # Permute for Conv1d: (batch_size, embed_dim, sequence_length)
    embedded = embedded.permute(0, 2, 1)

    conved = [F.relu(conv(embedded)) for conv in self.conv_layers]
    pooled = [F.adaptive_max_pool1d(conv, 1).squeeze(2) for conv in conved]
    cat = torch.cat(pooled, dim=1)
    return self.fc(cat)

class AudioFeatureExtractor(nn.Module):
  def __init__(self,input_channels,audio_hidden_dim,num_filters,kernel_sizes):
    super(AudioFeatureExtractor,self).__init__()
    self.conv_layers=nn.ModuleList([
        nn.Conv1d(input_channels,num_filters,k)for k in kernel_sizes
    ])
    self.fc=nn.Linear(len(kernel_sizes)*num_filters,audio_hidden_dim)
  def forward(self,x):
    conved=[F.relu(conv(x))for conv in self.conv_layers]
    pooled=[F.adaptive_max_pool1d(conv,1).squeeze(2)for conv in conved]
    cat=torch.cat(pooled,dim=1)
    return self.fc(cat)
  
class VisualFeatureExtractor(nn.Module):
  def __init__(self,in_channels,visual_hidden_dim,num_filters,kernel_sizes,image_width):
    super(VisualFeatureExtractor,self).__init__()
    self.conv_layers=nn.ModuleList([
        nn.Conv2d(in_channels,num_filters,k,padding='same')for k in kernel_sizes
    ])
    self.adaptive_pool=nn.AdaptiveAvgPool2d((1,1))
    self.fc=nn.Linear(len(kernel_sizes)*num_filters,visual_hidden_dim)
  def forward(self,x):
    conved=[F.relu(conv(x))for conv in self.conv_layers]
    pooled_and_flattend=[]
    for conv_out in conved:
      pooled=self.adaptive_pool(conv_out)
      pooled_and_flattend.append(pooled.squeeze(3).squeeze(2))
    cat=torch.cat(pooled_and_flattend,dim=1)
    return self.fc(cat)
  
class TensorFusionLayer(nn.Module):
  def __init__(self):
    super(TensorFusionLayer,self).__init__()
  def forward(self,h_text,h_audio,h_visual):
    batch_size=h_text.size(0)
    ones=torch.ones(batch_size,1,device=h_text.device)
    h_text_augmented=torch.cat([h_text,ones],dim=1)
    h_audio_augmented=torch.cat([h_audio,ones],dim=1)
    h_visual_augmented=torch.cat([h_visual,ones],dim=1)
    fused_tensor=torch.einsum('bi,bj,bk->bijk',h_text_augmented,h_audio_augmented,h_visual_augmented)
    flattened_fused_tensor=fused_tensor.view(batch_size,-1)
    return flattened_fused_tensor
  
class PredictionHead(nn.Module):
  def __init__(self,fused_dim,num_classes,task_type):
    super(PredictionHead,self).__init__()
    if task_type not in ['classification','regression']:
      raise ValueError("task_type must be 'classification' or 'regression'")
    self.task_type=task_type
    self.num_classes=num_classes
    self.fc1=nn.Linear(fused_dim,fused_dim//2)
    self.relu=nn.ReLU()
    if self.task_type =='classification':
      self.output_layer=nn.Linear(fused_dim//2,num_classes)
    else:
      self.output_layer=nn.Linear(fused_dim//2,1)
  def forward(self,fused_tensor):
    x=self.fc1(fused_tensor)
    x=self.relu(x)
    output=self.output_layer(x)
    if self.task_type=='classification':
      if self.num_classes>1:
        return F.softmax(output,dim=1)
      else:
        return torch.sigmoid(output)
    else:
      return output