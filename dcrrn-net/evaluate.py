import torch

def evaluate_model(model, data_loader, criterion, diffusion_f, diffusion_b,device=torch.device("cpu")):
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad(): # Disable gradient calculation during evaluation
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_X, diffusion_f, diffusion_b)
            loss = criterion(predictions, batch_y)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss
