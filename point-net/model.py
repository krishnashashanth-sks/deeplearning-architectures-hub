from layers import *

class PointNet2Classification(nn.Module):
  def __init__(self,num_classes):
    super(PointNet2Classification,self).__init__()
    # Fix: Missing '=' for nsample_list
    # Fix: Correct in_channel based on feature propagation (initial features are xyz)
    self.sa1=PointNetSetAbstractionMsg(
        npoint=512,radius_list=[0.1,0.2,0.4],nsample_list=[16,32,128],
        in_channel=0,mlp_list=[[32,32,64],[64,64,128],[64,96,128]] # in_channel=0 since only xyz is input as point features for sa1
    )
    # Output channels from sa1: 64+128+128 = 320
    # Fix: Consistent parameter name n_point -> npoint
    # Fix: Correct in_channel for sa2 to match sa1 output features
    self.sa2=PointNetSetAbstractionMsg(
        npoint=128,radius_list=[0.2,0.4,0.8],nsample_list=[32,64,128],
        in_channel=320,mlp_list=[[64,64,128],[128,128,256],[128,128,256]]
    )
    # Output channels from sa2: 128+256+256 = 640
    # Fix: Correct in_channel for sa3 to match sa2 output features
    self.sa3=PointNetSetAbstraction(
        npoint=None,radius=None,nsample=None,
        in_channel=640,mlp=[256,512,1024],
        group_all=True
    )

    self.fc1=nn.Linear(1024,512)
    # Fix: BatchNorm2d -> BatchNorm1d for features (B, C) after squeeze
    self.bn1=nn.BatchNorm1d(512)
    self.dropout1=nn.Dropout(0.4)
    self.fc2=nn.Linear(512,256)
    self.bn2=nn.BatchNorm1d(256)
    self.dropout2=nn.Dropout(0.4)
    self.fc3=nn.Linear(256,num_classes)

  def forward(self,x):
    # x is a Data object from PyTorch Geometric DataLoader
    xyz = x.pos # (Total_N, 3)
    batch = x.batch # (Total_N,)

    batch_size = batch.max().item()+1 if batch.numel()>0 else 1
    num_points_per_batch = xyz.shape[0]//batch_size
    xyz_reshaped = xyz.view(batch_size, num_points_per_batch, 3) # (B, N, 3)

    # Fix: Correct sequential calls to SA layers and feature passing
    # Initial features are just xyz for sa1
    l1_xyz, l1_features = self.sa1(xyz_reshaped, None) # l1_features: (B, C1_total, npoint1)

    # Pass l1_xyz and l1_features (permuted) to sa2
    # Features need to be (B, N, C) for SA layers if permuting in forward, or (B, C, N) if that's what's expected by SA
    # The SA layers expect (B, N_points, C) for features. So l1_features (B, C, N) needs to be permuted to (B, N, C).
    l2_xyz, l2_features = self.sa2(l1_xyz, l1_features.permute(0, 2, 1)) # l2_features: (B, C2_total, npoint2)

    # Pass l2_xyz and l2_features (permuted) to sa3
    l3_xyz, l3_features = self.sa3(l2_xyz, l2_features.permute(0, 2, 1)) # l3_features: (B, C3_total, 1)

    # Classifier head
    x = l3_features.squeeze(-1) # (B, C3_total) -> (B, 1024)

    # Fix: Ensure BatchNorm expects (B, C)
    x = F.relu(self.bn1(self.fc1(x)))
    x = self.dropout1(x)
    x = F.relu(self.bn2(self.fc2(x)))
    x = self.dropout2(x)
    x = self.fc3(x)
    return F.log_softmax(x,dim=-1)