import torch
import torchvision.transforms as transforms

#  Define a custom transformation class for self-supervised tasks (rotation prediction)
class RotationTransform:
    def __call__(self, img):
        # Randomly choose one of the four rotation angles
        rotation_idx = torch.randint(0, 4, (1,)).item()
        rotation_angle = rotation_idx * 90

        # Apply the rotation transformation
        rotated_img = transforms.functional.rotate(img, rotation_angle)
        rotation_label = rotation_idx

        return rotated_img, rotation_label