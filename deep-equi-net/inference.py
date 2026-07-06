import torch

def predict_image(model, image_tensor, device, class_names):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations
        image_tensor = image_tensor.to(device) # Move image to device
        outputs = model(image_tensor.unsqueeze(0))  # Add batch dimension and pass through model
        _, predicted = torch.max(outputs.data, 1) # Get the predicted class
    return class_names[predicted.item()]