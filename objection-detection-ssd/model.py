import torch.nn as nn
from  layers import SSDHead
import torchvision.models as models
import torch

class AdvancedSSD(nn.Module):
  def __init__(self,backbone_name='resnet50',num_classes=21,num_anchors_per_location=6):
    super().__init__()
    self.num_classes=num_classes
    self.num_anchors_per_location=num_anchors_per_location
    if backbone_name=='resnet50':
      self.backbone=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
      self.feature_extractor=nn.Sequential(*list(self.backbone.children())[:-2]) # Corrected: removed 's'
      backbone_out_channels=2048
    elif backbone_name=='mobilenet_v3_large':
      self.backbone=models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
      self.feature_extractor=self.backbone.features
      backbone_out_channels=960
    else:
      raise ValueError(f"Backbone '{backbone_name}' not supported. Choose 'resnet50' or 'mobilenet_v3_large'.")
    self.extra_layers=nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(backbone_out_channels,512,kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(512,1024,kernel_size=3,stride=2,padding=1),
        nn.ReLU()
      ),
      nn.Sequential(
          nn.Conv2d(1024,256,kernel_size=1),
          nn.ReLU(),
          nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1),
          nn.ReLU() # Added missing ReLU
          ),
          nn.Sequential(
              nn.Conv2d(512,128,kernel_size=1),
              nn.ReLU(),
              nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
              nn.ReLU()
          )    ,
          nn.Sequential(
              nn.Conv2d(256,128,kernel_size=1),
              nn.ReLU(),
              nn.Conv2d(128,256,kernel_size=3,padding=1), # Changed padding from 0 to 1
              nn.ReLU()
          )
    ])
    if backbone_name=='resnet50':
      feature_map_channels=[backbone_out_channels,1024,512,256,256]
    elif backbone_name=='mobilenet_v3_large':
      feature_map_channels=[backbone_out_channels,1024,512,256,256]
    self.prediction_heads=nn.ModuleList()
    for in_channels in feature_map_channels:
      self.prediction_heads.append(SSDHead(in_channels,self.num_anchors_per_location,num_classes))
  def forward(self,x):
    feature_maps=[]
    x=self.feature_extractor(x)
    feature_maps.append(x) # Append initial backbone feature map

    # Iterate through extra layers and append each generated feature map
    for layer in self.extra_layers:
      x=layer(x)
      feature_maps.append(x)

    cls_predictions=[]
    reg_predictions=[]
    for i,fm in enumerate(feature_maps):
      cls_pred,reg_pred=self.prediction_heads[i](fm)
      cls_pred=cls_pred.permute(0,2,3,1).contiguous()
      reg_pred=reg_pred.permute(0,2,3,1).contiguous()
      cls_predictions.append(cls_pred.view(cls_pred.size(0),-1,self.num_classes))
      reg_predictions.append(reg_pred.view(reg_pred.size(0),-1,4))
    cls_output=torch.cat(cls_predictions,dim=1)
    reg_output=torch.cat(reg_predictions,dim=1)
    return cls_output,reg_output
