import torch

def predict_model(model, data_loader, diffusion_f, diffusion_b,device=torch.device("cpu")):
    model.eval()  # Set model to evaluation mode
    all_predictions = []

    with torch.no_grad():  # Disable gradient calculation during inference
        for batch_X, _ in data_loader:  # We only need batch_X for prediction
            batch_X = batch_X.to(device)

            predictions = model(batch_X, diffusion_f, diffusion_b)
            all_predictions.append(predictions.cpu())  # Move predictions to CPU and store

    # Concatenate all predictions along the batch dimension
    return torch.cat(all_predictions, dim=0)
