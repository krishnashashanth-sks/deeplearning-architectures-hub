import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
  def __init__(self,in_channels,num_filters):
    super(Encoder,self).__init__()
    self.block1=self._conv_block(in_channels,num_filters)
    self.block2=self._conv_block(num_filters,num_filters*2)
    self.block3=self._conv_block(num_filters*2,num_filters*4)
  def _conv_block(self,in_c,out_c):
    return nn.Sequential(
        nn.Conv2d(in_c,out_c,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True),
        nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
  def forward(self,x):
    return self.block3(self.block2(self.block1(x)))

class Decoder(nn.Module):
  def __init__(self,num_filters):
    super(Decoder,self).__init__()
    self.up1=self._upconv_block(num_filters*4,num_filters*2)
    self.up2=self._upconv_block(num_filters*2,num_filters)
    self.final_conv=nn.Conv2d(num_filters,num_filters//2,kernel_size=1)
  def _upconv_block(self,in_c,out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c,out_c,kernel_size=2,stride=2),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True),
        nn.ConvTranspose2d(out_c,out_c,kernel_size=2,stride=2),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True),
    )
  def forward(self,x):
    return self.final_conv(self.up2(self.up1(x)))
  
class KeypointDetector(nn.Module):
  def __init__(self,in_channels,num_keypoints,num_filters=32):
    super(KeypointDetector,self).__init__()
    self.encoder=Encoder(in_channels,num_filters)
    self.decoder=Decoder(num_filters)
    self.num_keypoints=num_keypoints
    self.heatmap_head=nn.Conv2d(num_filters//2,num_keypoints,kernel_size=1)
    self.kp_feature_extractor=nn.AdaptiveAvgPool2d((1,1))
    self.jacobian_head=nn.Linear(num_filters//2,num_keypoints*6)
  def forward(self,x):
    encoded_features=self.encoder(x)
    decoded_features=self.decoder(encoded_features)
    heatmaps=self.heatmap_head(decoded_features)
    global_features=self.kp_feature_extractor(decoded_features).squeeze()
    if global_features.dim()==1:
      global_features=global_features.unsqueeze(0)
    jacobians_flat=self.jacobian_head(global_features)
    jacobians=jacobians_flat.view(-1,self.num_keypoints,2,3)
    return {
            'heatmaps': heatmaps,
            'jacobians': jacobians
        }
  
# Helper for Convolutional blocks
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

# Helper for Upsampling and Convolutional blocks
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels) # Use ConvBlock after upsampling
    def forward(self, x):
        return self.conv(self.upsample(x))

class DenseMotionNetwork(nn.Module):
  def __init__(self,in_channels,num_keypoints,num_filters=32):
    super(DenseMotionNetwork,self).__init__()
    input_for_first_conv=in_channels+num_keypoints*2
    self.initial_conv=ConvBlock(input_for_first_conv,num_filters)
    self.encoder1=ConvBlock(num_filters,num_filters*2,stride=2)
    self.encoder2=ConvBlock(num_filters*2,num_filters*4,stride=2)
    self.encoder3=ConvBlock(num_filters*4,num_filters*8,stride=2)
    self.decoder3=UpConvBlock(num_filters*8,num_filters*4)
    self.decoder_conv3=ConvBlock(num_filters*4+num_filters*4,num_filters*4)
    self.decoder2=UpConvBlock(num_filters*4,num_filters*2)
    self.decoder_conv2=ConvBlock(num_filters*2+num_filters*2,num_filters*2)
    self.decoder1=UpConvBlock(num_filters*2,num_filters)
    self.decoder_conv1=ConvBlock(num_filters+num_filters,num_filters)
    self.motion_head=nn.Conv2d(num_filters,2,kernel_size=3,padding=1)
    self.occlusion_head=nn.Conv2d(num_filters,1,kernel_size=3,padding=1)
  def forward(self,source_image,source_kp,driving_kp):
    source_heatmaps=source_kp['heatmaps']
    driving_heatmaps=source_kp['heatmaps']
    if source_heatmaps.shape[-1:]!=source_image.shape[-2:]:
      source_heatmaps=F.interpolate(source_heatmaps,size=source_image.shape[-2:],mode='bilinear',align_corners=False)
      driving_heatmaps=F.interpolate(driving_heatmaps,size=source_image.shape[-2:],mode='bilinear',align_corners=False)
    motion_context=torch.cat([source_heatmaps,driving_heatmaps],dim=1)
    x=torch.cat([source_image,motion_context],dim=1)
    e0=self.initial_conv(x)
    e1=self.encoder1(e0)
    e2=self.encoder2(e1)
    e3=self.encoder3(e2)
    d3=self.decoder3(e3)
    d3=torch.cat([d3,e2],dim=1)
    d3=self.decoder_conv3(d3)
    d2=self.decoder2(d3)
    d2=torch.cat([d2,e1],dim=1)
    d2=self.decoder_conv2(d2)
    d1=self.decoder1(d2)
    d1=torch.cat([d1,e0],dim=1)
    d1=self.decoder_conv1(d1)
    dense_motion_field=self.motion_head(d1)
    occlusion_map=torch.sigmoid(self.occlusion_head(d1))
    return {
        "dense_motion_field":dense_motion_field,
        "occlusion_map":occlusion_map
    }
  
class OcclusionAwareGenerator(nn.Module):
  def __init__(self,in_channels,num_filters=32):
    super(OcclusionAwareGenerator,self).__init__()
    input_for_first_conv=in_channels+2+1
    self.initial_conv=ConvBlock(input_for_first_conv,num_filters)
    self.encoder1=ConvBlock(num_filters,num_filters*2,stride=2)
    self.encoder2=ConvBlock(num_filters*2,num_filters*4,stride=2)
    self.encoder3=ConvBlock(num_filters*4,num_filters*8,stride=2)
    self.decoder3=UpConvBlock(num_filters*8,num_filters*4)
    self.decoder_conv3=ConvBlock(num_filters*4+num_filters*4,num_filters*4)

    self.decoder2=UpConvBlock(num_filters*4,num_filters*2)
    self.decoder_conv2=ConvBlock(num_filters*2+num_filters*2,num_filters*2)

    self.decoder1=UpConvBlock(num_filters*2,num_filters)
    self.decoder_conv1=ConvBlock(num_filters+num_filters,num_filters)
    self.final_conv=nn.Conv2d(num_filters,in_channels,kernel_size=3,padding=1)
  def forward(self,source_image,dense_motion_field,occlusion_map):
    grid_h,grid_w=source_image.shape[2:]
    identity_grid=torch.meshgrid(
      torch.linspace(-1,1,grid_w,device=source_image.device),
      torch.linspace(-1,1,grid_h,device=source_image.device)
    )
    identity_grid=torch.stack(identity_grid,dim=2)
    identity_grid=identity_grid.unsqueeze(0).repeat(source_image.shape[0],1,1,1)
    norm_motion_field_x=2*(dense_motion_field[:,0,:,:]/(grid_w-1))
    norm_motion_field_y=2*(dense_motion_field[:,1,:,:]/(grid_h-1))
    norm_motion_field=torch.stack([norm_motion_field_x,norm_motion_field_y],dim=3)
    warped_grid=identity_grid+norm_motion_field
    warped_grid=F.grid_sample(source_image,warped_grid,mode='bilinear',padding_mode='reflection',align_corners=False)
    x=torch.cat([warped_grid,dense_motion_field,occlusion_map],dim=1)
    e0=self.initial_conv(x)
    e1=self.encoder1(e0)
    e2=self.encoder2(e1)
    e3=self.encoder3(e2)
    d3=self.decoder3(e3)
    d3=torch.cat([d3,e2],dim=1)
    d3=self.decoder_conv3(d3)
    d2=self.decoder2(d3)
    d2=torch.cat([d2,e1],dim=1)
    d2=self.decoder_conv2(d2)
    d1=self.decoder1(d2)
    d1=torch.cat([d1,e0],dim=1)
    d1=self.decoder_conv1(d1)
    generated_image=torch.sigmoid(self.final_conv(d1))
    return generated_image