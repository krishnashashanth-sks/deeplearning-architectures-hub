import torch.nn as nn
from layers import KeypointDetector,DenseMotionNetwork,OcclusionAwareGenerator

class FirstOrderMotionModel(nn.Module):
  def __init__(self,in_channels,num_keypoints,num_filters=32):
    super(FirstOrderMotionModel,self).__init__()
    self.keypoint_detector=KeypointDetector(in_channels,num_keypoints,num_filters)
    self.dense_motion_network=DenseMotionNetwork(in_channels,num_keypoints,num_filters)
    self.occlusion_aware_generator=OcclusionAwareGenerator(in_channels,num_filters)
  def forward(self,source_image,driving_image):
    source_kp_output=self.keypoint_detector(source_image)
    driving_kp_output=self.keypoint_detector(driving_image)
    motion_output=self.dense_motion_network(
        source_image,source_kp_output,driving_kp_output
    )
    dense_motion_field=motion_output['dense_motion_field']
    occlusion_map=motion_output['occlusion_map']
    generated_image=self.occlusion_aware_generator(
        source_image,dense_motion_field,occlusion_map
    )
    return {
            'generated_image': generated_image,
            'dense_motion_field': dense_motion_field,
            'occlusion_map': occlusion_map,
            'source_kp_output': source_kp_output, # Optionally return for debugging/loss
            'driving_kp_output': driving_kp_output # Optionally return for debugging/loss
        }