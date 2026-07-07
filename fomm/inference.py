import torch

def inference(model, source_image, driving_image, device):
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        source_image = source_image.to(device).unsqueeze(0) # Add batch dimension and move to device
        driving_image = driving_image.to(device).unsqueeze(0) # Add batch dimension and move to device

        output = model(source_image, driving_image)
        generated_image = output['generated_image'].squeeze(0) # Remove batch dimension

    return generated_image.cpu()
