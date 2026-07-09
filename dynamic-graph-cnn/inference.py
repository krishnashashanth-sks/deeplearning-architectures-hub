import torch

def predict(model, point_cloud_data, device):
    """
    Performs inference on a single point cloud or a batch of point clouds.

    Args:
        model (nn.Module): The trained DGCNN model.
        point_cloud_data (torch.Tensor): Input point cloud data.
                                         Expected shape: (num_points, in_channels) for a single sample,
                                         or (batch_size, num_points, in_channels) for a batch.
        device (torch.device): The device (CPU or GPU) to perform inference on.

    Returns:
        tuple: A tuple containing:
            - logits (torch.Tensor): The raw output logits from the model.
            - predicted_class (torch.Tensor): The predicted class index(es).
    """
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation during inference
        # Move input data to the specified device
        point_cloud_data = point_cloud_data.to(device)

        # The DGCNN model's forward pass expects input of shape (batch_size, num_points, in_channels).
        # If a single point cloud is provided (2D tensor), add a batch dimension.
        if point_cloud_data.dim() == 2:
            point_cloud_data = point_cloud_data.unsqueeze(0) # Adds a batch dimension: (1, num_points, in_channels)

        # Perform forward pass
        logits = model(point_cloud_data)

        # Get the predicted class index
        _, predicted_class = torch.max(logits, 1)

        return logits, predicted_class
