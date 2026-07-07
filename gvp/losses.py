import torch.nn as nn

class GVPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_s, pred_v, target_s, target_v):
        loss_s = self.mse(pred_s, target_s)
        # Ensure vector comparison is handled correctly (e.g., flatten or compare element-wise)
        loss_v = self.mse(pred_v.flatten(), target_v.flatten())
        return loss_s + loss_v