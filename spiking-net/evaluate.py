import torch
from snntorch import functional as SF

def evaluate_snn(model, test_loader, num_steps, device):
    print("Starting SNN evaluation...")
    model.eval() # Set model to evaluation mode

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # Disable gradient calculations
        for data, targets in test_loader:
            data = data.to(device) # Move input data to device
            targets = targets.to(device) # Move target labels to device

            # Forward pass
            spk_rec = model(data, num_steps)

            # Calculate accuracy for the current batch
            batch_acc = SF.accuracy_rate(spk_rec, targets)
            correct_predictions += batch_acc * len(targets)
            total_samples += len(targets)

    test_accuracy = (correct_predictions / total_samples) * 100
    print(f"SNN evaluation finished. Test Accuracy: {test_accuracy:.2f}%")
    return test_accuracy