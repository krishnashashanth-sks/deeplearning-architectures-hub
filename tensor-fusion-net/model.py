import torch.nn as nn
from layers import TensorFusionLayer,TextFeatureExtractor,AudioFeatureExtractor,VisualFeatureExtractor,PredictionHead

class TFNModel(nn.Module):
  def __init__(self,vocab_size,embed_dim,text_hidden_dim,num_filters_text,kernel_sizes_text,
               audio_feature_dim,audio_hidden_dim,num_filters_audio,kernel_sizes_audio,
               visual_channels,visual_hidden_dim,num_filters_visual,kernel_sizes_visual,image_height,image_width,num_classes,task_type):
    super(TFNModel,self).__init__()
    self.text_fe=TextFeatureExtractor(vocab_size,embed_dim,text_hidden_dim,num_filters_text,kernel_sizes_text)
    self.audio_fe=AudioFeatureExtractor(audio_feature_dim, audio_hidden_dim, num_filters_audio, kernel_sizes_audio)
    self.visual_fe=VisualFeatureExtractor(visual_channels,visual_hidden_dim,num_filters_visual,kernel_sizes_visual,image_width)
    self.fusion_layer=TensorFusionLayer()
    fused_dim=(text_hidden_dim+1)*(audio_hidden_dim+1)*(visual_hidden_dim+1)
    self.prediction_head=PredictionHead(fused_dim,num_classes,task_type)
  def forward(self,text_input,audio_input,visual_input):
    h_text=self.text_fe(text_input)
    h_audio=self.audio_fe(audio_input)
    h_visual=self.visual_fe(visual_input)
    fused_tensor=self.fusion_layer(h_text,h_audio,h_visual)
    return self.prediction_head(fused_tensor)