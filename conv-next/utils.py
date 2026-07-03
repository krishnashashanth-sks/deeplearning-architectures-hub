import matplotlib.pyplot as plt
import numpy as np

# Function to unnormalize and display an image
def imshow(img):
    # Unnormalize for CIFAR-10
    # Mean: (0.4914, 0.4822, 0.4465), Std: (0.2023, 0.1994, 0.2010)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = std * img.numpy().transpose((1, 2, 0)) + mean # unnormalize
    img = np.clip(img, 0, 1) # Clip values to [0, 1] range
    plt.imshow(img)
    plt.axis('off')
    
def imshow_unnormalize(img_tensor):
    # Unnormalize for CIFAR-10
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = img_tensor.cpu().numpy().transpose((1, 2, 0)) # Move to CPU, then to numpy, then transpose
    img = std * img + mean # unnormalize
    img = np.clip(img, 0, 1) # Clip values to [0, 1] range
    plt.imshow(img)
    plt.axis('off')