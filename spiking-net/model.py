import torch.nn as nn
import torch
import snntorch as snn
from snntorch import surrogate

class Net(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta, threshold):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(),
            reset_mechanism="zero"
        )
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(),
            reset_mechanism="zero"
        )

    def forward(self, x, num_steps):
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record output spikes
        spk_rec = []

        for step_idx in range(num_steps):
            # Flatten input for the first linear layer
            cur = self.fc1(x.flatten(1))

            # Pass through first LIF layer
            spk1, mem1 = self.lif1(cur, mem1)

            # Pass through second linear layer
            cur = self.fc2(spk1)

            # Pass through second LIF layer
            spk2, mem2 = self.lif2(cur, mem2)

            spk_rec.append(spk2)

        # Stack recorded spikes
        return torch.stack(spk_rec)