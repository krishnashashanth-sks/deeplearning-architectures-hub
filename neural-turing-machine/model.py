import torch.nn as nn
from layers import Controller,Memory,ReadHead,WriteHead
import torch

# Full NTM Model
class NTM(nn.Module):
    def __init__(self, input_size, output_size, mem_size, mem_vector_size,
                 num_read_heads, num_write_heads, controller_hidden_size):
        super(NTM, self).__init__()
        self.mem_size = mem_size
        self.mem_vector_size = mem_vector_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.controller_hidden_size = controller_hidden_size

        self.controller = Controller(input_size, controller_hidden_size,
                                     num_read_heads, mem_vector_size, output_size)
        self.memory = Memory(mem_size, mem_vector_size)

        self.read_heads = nn.ModuleList([ReadHead(mem_size, mem_vector_size, controller_hidden_size) for _ in range(num_read_heads)])
        self.write_heads = nn.ModuleList([WriteHead(mem_size, mem_vector_size, controller_hidden_size) for _ in range(num_write_heads)])

    def reset(self, batch_size, device):
        self.memory.reset(batch_size, device)
        self.controller_state = (torch.zeros(batch_size, self.controller_hidden_size).to(device),
                                 torch.zeros(batch_size, self.controller_hidden_size).to(device))
        self.prev_read_weights = [torch.zeros(batch_size, self.mem_size).fill_(1/self.mem_size).to(device) for _ in range(self.num_read_heads)]
        self.prev_write_weights = [torch.zeros(batch_size, self.mem_size).fill_(1/self.mem_size).to(device) for _ in range(self.num_write_heads)]

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        device = x.device
        self.reset(batch_size, device)

        outputs = []
        current_read_vectors = [torch.zeros(batch_size, self.mem_vector_size).to(device) for _ in range(self.num_read_heads)]

        for t in range(seq_len):
            current_input = x[:, t, :]

            model_output, head_params, self.controller_state = self.controller(current_input, self.controller_state, current_read_vectors)

            current_read_vectors = []
            mem_snapshot = self.memory()
            for i, head in enumerate(self.read_heads):
                rv, rw = head(head_params, mem_snapshot, self.prev_read_weights[i])
                current_read_vectors.append(rv)
                self.prev_read_weights[i] = rw

            for i, head in enumerate(self.write_heads):
                updated_mem, ww = head(head_params, mem_snapshot, self.prev_write_weights[i])
                mem_snapshot = updated_mem
                self.prev_write_weights[i] = ww

            self.memory.update_memory(mem_snapshot)
            outputs.append(model_output)

        return torch.stack(outputs, dim=1)
