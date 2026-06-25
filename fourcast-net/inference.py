import torch

def inference_function(model, dataloader, device):
    model.eval() # Set the model to evaluation mode
    predictions = []
    true_targets = []
    input_data = []

    with torch.no_grad(): # Disable gradient calculations during inference
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            input_data.append(inputs.cpu())
            predictions.append(outputs.cpu())
            true_targets.append(targets.cpu())

    # Concatenate all batches
    input_data = torch.cat(input_data, dim=0)
    predictions = torch.cat(predictions, dim=0)
    true_targets = torch.cat(true_targets, dim=0)

    return input_data, true_targets, predictions