import os
import torch
import numpy as np
from monai.transforms import(
    AsDiscrete,
    Compose,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    ScaleIntensityRanged # Added ScaleIntensityRanged
)
from monai.data import decollate_batch,DataLoader,Dataset
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
import matplotlib.pyplot as plt
from train import train_model

# Set deterministic training for reproducibility
set_determinism(seed=0);

# Create dummy data for demonstration
# In a real scenario, you'd have actual image and label paths
dummy_data_dir = "./dummy_data"
os.makedirs(dummy_data_dir, exist_ok=True);

num_samples = 5
image_size = (96, 96, 96);

train_images = []
train_labels = [];

for i in range(num_samples):
    # Create a dummy image (e.g., random noise)
    dummy_image = np.random.rand(*image_size).astype(np.float32)
    img_path = os.path.join(dummy_data_dir, f"image_{i:03d}.npy")
    np.save(img_path, dummy_image)
    train_images.append(img_path);

    # Create a dummy label (e.g., random binary mask)
    dummy_label = (np.random.rand(*image_size) > 0.8).astype(np.float32)
    label_path = os.path.join(dummy_data_dir, f"label_{i:03d}.npy")
    np.save(label_path, dummy_label)
    train_labels.append(label_path);

train_files = [
    {"image": img, "label": lbl}
    for img, lbl in zip(train_images, train_labels)
];

# Define the training transformations
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]), # Load image and label from specified keys
        EnsureChannelFirstd(keys=["image", "label"]), # Ensure channel dimension is first
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57, a_max=164,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1, neg=1, num_samples=4,
            allow_smaller=True # Added to prevent cropping errors
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
);

# Create a MONAI Dataset and DataLoader
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0);

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the UNet model
# No pre-trained weights are used; the model is initialized randomly.
model = UNet(
    spatial_dims=3, # 3D U-Net
    in_channels=1,  # Number of input channels (e.g., 1 for grayscale medical images)
    out_channels=2, # Number of output channels (e.g., 2 for foreground/background segmentation)
    channels=(16, 32, 64, 128, 256), # Number of feature maps for each layer
    strides=(2, 2, 2, 2), # Strides for downsampling
    num_res_units=2, # Number of residual units
    norm=Norm.BATCH, # Normalization layer
).to(device)

print("U-Net model defined successfully, without pre-trained weights.")

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)]) # Post-processing for predictions
post_label = Compose([AsDiscrete(to_onehot=2)]) # Post-processing for labels
max_epochs = 5
epoch_loss_values= train_model(max_epochs,model,train_loader,optimizer,loss_function,train_ds,device)


# Plot the training loss
plt.figure(figsize=(8, 6))
plt.plot(epoch_loss_values)
plt.title("Epoch Average Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

model.eval()
with torch.no_grad():
    # For demonstration, we'll take one batch from the training loader
    # In a real scenario, you'd use a separate validation/test loader
    sample_batch = next(iter(train_loader))
    val_inputs = sample_batch["image"].to(device)
    val_labels = sample_batch["label"].to(device)

    roi_size = (96, 96, 96) # Region of Interest for sliding window
    sw_batch_size = 4 # Batch size for sliding window inference

    val_outputs = sliding_window_inference(
        val_inputs, roi_size, sw_batch_size, model
    )

    # Post-process the outputs to get the final segmentation map
    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
    val_labels = [post_label(i) for i in decollate_batch(val_labels)]

    # Calculate Dice metric
    dice_metric(y_pred=val_outputs, y=val_labels)
    metric = dice_metric.aggregate().item()
    dice_metric.reset()

    print(f"Inference completed. Example Dice Metric: {metric:.4f}")

    # Visualize an example slice (e.g., from the first image in the batch)
    plt.figure("inference", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, 24], cmap="gray") # Display a central slice (index 24 for 48 depth)
    plt.subplot(1, 2, 2)
    plt.title("Predicted Segmentation")
    plt.imshow(val_outputs[0].cpu().numpy()[1, :, :, 24], cmap="jet", alpha=0.5) # Display foreground class (index 24 for 48 depth)
    plt.show()