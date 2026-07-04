import torch
from utils import calculate_accuracy

def evaluate_model(test_loader,model,criterion,device):
    test_running_loss = 0.0
    test_running_corrects = 0
    test_total_samples = 0
    all_predicted_labels = []
    all_true_labels = []

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_running_loss += loss.item() * inputs.size(0)
            test_running_corrects += calculate_accuracy(outputs, labels) * inputs.size(0)
            test_total_samples += inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            all_predicted_labels.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    return test_running_loss,test_running_corrects,test_total_samples,all_true_labels,all_predicted_labels
