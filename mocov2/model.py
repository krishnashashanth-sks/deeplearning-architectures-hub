import torch.nn as nn
import torch
from layers import ProjectionHead
import torch.nn.functional as F

# MoCo Architecture
class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp_hidden_dim=2048):
        super(MoCo, self).__init__()

        self.K = K          # Queue size
        self.m = m          # Momentum coefficient
        self.T = T          # Temperature for InfoNCE loss

        # Query encoder and projector
        self.encoder_q = base_encoder()
        self.projector_q = ProjectionHead(
            in_features=self.encoder_q.output_dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim
        )

        # Key encoder and projector (no gradients, updated via momentum)
        self.encoder_k = base_encoder()
        self.projector_k = ProjectionHead(
            in_features=self.encoder_k.output_dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim
        )

        # Initialize key encoder with query encoder's weights and freeze them
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Register the memory bank (queue) as a buffer
        self.register_buffer("queue", torch.randn(dim, K)) # (dim, K)
        self.queue = F.normalize(self.queue, dim=0)

        # Register a pointer for the queue
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Dequeue oldest batch and enqueue new keys"""
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # Ensure queue can be filled by whole batches
        assert self.K % batch_size == 0

        # Replace the oldest batch of keys with the new ones
        self.queue[:, ptr:ptr + batch_size] = keys.T # Transpose to (dim, batch_size)
        ptr = (ptr + batch_size) % self.K  # Move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """ Forward pass for MoCo
        Args:
            im_q (Tensor): query images
            im_k (Tensor): key images
        Returns:
            logits (Tensor): logits for InfoNCE loss
            labels (Tensor): ground-truth labels for InfoNCE loss
        """
        # Encode query
        q = self.projector_q(self.encoder_q(im_q))
        q = F.normalize(q, dim=1)

        # Encode key with no grad and momentum update
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.projector_k(self.encoder_k(im_k))
            k = F.normalize(k, dim=1)

        # Compute logits
        # Positive logits: q * k
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # Negative logits: q * queue
        # Detach queue from graph, as it's not trained via backprop
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # Concatenate positive and negative logits
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature
        logits /= self.T

        # Labels: first column (l_pos) corresponds to the positive sample
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # Update the queue
        self._dequeue_and_enqueue(k)

        return logits, labels

