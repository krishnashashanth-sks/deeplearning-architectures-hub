import torch

def train(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        src = batch['nl_input'].to(device)
        trg = batch['code_target'].to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)
