import torch.nn.functional as F

def train_sage(data,graphsage_model,optimizer_sage):
  graphsage_model.train()
  optimizer_sage.zero_grad()
  out=graphsage_model(data)
  loss=F.nll_loss(out[data.train_mask],data.y[data.train_mask])
  loss.backward()
  optimizer_sage.step()
  return loss.item()
def test_sage(data,graphsage_model):
  graphsage_model.eval()
  out=graphsage_model(data)
  pred=out.argmax(dim=1)
  correct=(pred[data.test_mask]==data.y[data.test_mask]).sum()
  acc=int(correct)/int(data.test_mask.sum())
  return acc