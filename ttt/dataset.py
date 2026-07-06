import torchvision
import torchvision.transforms as transforms

class CIFAR10TestSSL(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False,
                 rotation_transform=None, post_rotation_transform=None):
        super().__init__(root, train, transform, target_transform, download)
        self.rotation_transform = rotation_transform
        self.post_rotation_transform = post_rotation_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = transforms.ToPILImage()(img)

        if self.transform is not None:
            # Apply standard test transformation (for classification head input)
            original_img = self.transform(img)
        else:
            original_img = transforms.ToTensor()(img) # Default to ToTensor if no transform

        # Apply self-supervised transformations
        ss_img = original_img # Default if no SS transform
        ss_label = 0 # Default if no SS transform

        if self.rotation_transform is not None:
            # First apply rotation to get rotated_img and rotation_label
            rotated_img_pil, ss_label = self.rotation_transform(img)
            # Then apply post-rotation transforms (ToTensor, Normalize) to the rotated image
            if self.post_rotation_transform is not None:
                ss_img = self.post_rotation_transform(rotated_img_pil)
            else:
                ss_img = transforms.ToTensor()(rotated_img_pil) # Default to ToTensor

        return original_img, target, ss_img, ss_label
