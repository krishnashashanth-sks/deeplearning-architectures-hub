import torch

def predict_sparse_code(model, y, device):
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient computation for inference
        y = y.to(device)
        x_pred = model(y)
    return x_pred.cpu() # Move predictions back to CPU if needed