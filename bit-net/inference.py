import torch

def inference_single_image(model, image_tensor, device):
    """
    Performs inference on a single image using the given model.

    Args:
        model (nn.Module): The trained model.
        image_tensor (torch.Tensor): The preprocessed image tensor (e.g., flattened MNIST image).
                                     Expected shape: (input_dim,) for single image.
        device (torch.device): The device (CPU or GPU) to run inference on.

    Returns:
        int: The predicted class label.
    """
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        # Add a batch dimension to the single image tensor
        image_tensor = image_tensor.unsqueeze(0).to(device) # Shape becomes (1, input_dim)
        output = model(image_tensor)
        # Get the predicted class by finding the index of the maximum log-probability
        prediction = output.argmax(dim=1, keepdim=True).item()
    return prediction
