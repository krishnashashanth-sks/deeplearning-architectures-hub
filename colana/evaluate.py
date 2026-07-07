import torch

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src = batch['nl_input'].to(device)
            trg = batch['code_target'].to(device)

            output = model(src, trg, 0) # Turn off teacher forcing during evaluation

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)