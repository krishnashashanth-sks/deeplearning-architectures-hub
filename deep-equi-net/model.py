import torch.nn as nn
from layers import DEQFixedPointFunc,DEQLayer

class DEQClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes, max_iter=50, tol=1e-4):
        super(DEQClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.initial_projection = nn.Linear(input_size, hidden_dim)
        self.initial_activation = nn.GELU()
        self.deq_func = DEQFixedPointFunc(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.deq_layer = DEQLayer(func=self.deq_func, max_iter=max_iter, tol=tol)
        self.classification_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x_projected = self.initial_activation(self.initial_projection(x))
        z_star = self.deq_layer(x_projected)
        logits = self.classification_head(z_star)
        return logits