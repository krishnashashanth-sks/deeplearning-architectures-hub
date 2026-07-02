import torch.nn.functional as F

def train(data,model,optimizer):
    model.train() # Set the model to training mode
    optimizer.zero_grad() # Clear gradients
    out = model(data) # Perform a single forward pass
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) # Calculate the negative log likelihood loss
    loss.backward() # Derive gradients
    optimizer.step() # Update model parameters
    return loss.item()

def test(data,model):
    model.eval() # Set the model to evaluation mode
    out = model(data) # Perform a single forward pass
    pred = out.argmax(dim=1) # Use the class with highest probability
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum() # Check against ground-truth labels
    acc = int(correct) / int(data.test_mask.sum()) # Calculate accuracy
    return acc