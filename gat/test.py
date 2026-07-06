import torch

def test(model,data):
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        out = model(data.x, data.edge_index) # Perform a forward pass

    # Get predicted class labels
    pred = out.argmax(dim=1) # Use the class with the highest probability

    # Calculate accuracy for each mask
    correct_train = pred[data.train_mask] == data.y[data.train_mask]
    train_acc = int(correct_train.sum()) / int(data.train_mask.sum())

    correct_val = pred[data.val_mask] == data.y[data.val_mask]
    val_acc = int(correct_val.sum()) / int(data.val_mask.sum())

    correct_test = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(correct_test.sum()) / int(data.test_mask.sum())

    return train_acc, val_acc, test_acc
