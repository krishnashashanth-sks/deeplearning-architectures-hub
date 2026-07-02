import torch.nn.functional as F

def train_gat(data,gat_model,optimizer_gat):
    gat_model.train()
    optimizer_gat.zero_grad()
    out = gat_model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer_gat.step()
    return loss.item()

def test_gat(gat_model,data):
    gat_model.eval()
    out = gat_model(data)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc
