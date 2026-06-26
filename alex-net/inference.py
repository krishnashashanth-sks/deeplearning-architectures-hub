from PIL import Image
import torch

def predict_image(model, image: Image.Image, transform, device, class_names):
    """
    Predicts the class of a single input image using the trained model.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        image (PIL.Image.Image): The input image to predict.
        transform (torchvision.transforms.Compose): The transformations to apply to the image.
        device (torch.device): The device (CPU or GPU) to run inference on.
        class_names (tuple): A tuple of class names.

    Returns:
        tuple: A tuple containing the predicted class name and the prediction probabilities.
    """
    model.eval()  # Set the model to evaluation mode

    # Apply transformations to the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # Get the predicted class index
        _, predicted_idx = torch.max(outputs.data, 1)

        predicted_class_name = class_names[predicted_idx.item()]

    return predicted_class_name, probabilities.cpu().numpy().flatten()
