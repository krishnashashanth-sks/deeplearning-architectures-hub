import torch.nn as nn
import torch
import torch.nn.functional as F

# 1. Memory Bank
class Memory(nn.Module):
    def __init__(self, mem_size, mem_vector_size):
        super(Memory, self).__init__()
        self.mem_size = mem_size
        self.mem_vector_size = mem_vector_size
        self._memory = None

    def reset(self, batch_size, device):
        self._memory = torch.zeros(batch_size, self.mem_size, self.mem_vector_size).to(device)

    def forward(self):
        return self._memory

    def update_memory(self, new_memory_state):
        self._memory = new_memory_state

# 2. Conceptual Read Head
class ReadHead(nn.Module):
    def __init__(self, mem_size, mem_vector_size, controller_output_size):
        super(ReadHead, self).__init__()
        self.mem_size = mem_size
        self.mem_vector_size = mem_vector_size
        self.key_layer = nn.Linear(controller_output_size, mem_vector_size)
        self.beta_layer = nn.Linear(controller_output_size, 1)

    def forward(self, controller_output, memory, prev_weights):
        key = torch.tanh(self.key_layer(controller_output))
        beta = F.softplus(self.beta_layer(controller_output))
        similarity = F.cosine_similarity(key.unsqueeze(1), memory, dim=2)
        read_weights = F.softmax(beta.unsqueeze(1) * similarity, dim=1)

        read_vector = torch.matmul(read_weights.unsqueeze(1), memory).view(memory.size(0), -1)
        return read_vector, read_weights

# 3. Conceptual Write Head
class WriteHead(nn.Module):
    def __init__(self, mem_size, mem_vector_size, controller_output_size):
        super(WriteHead, self).__init__()
        self.mem_size = mem_size
        self.mem_vector_size = mem_vector_size
        self.key_layer = nn.Linear(controller_output_size, mem_vector_size)
        self.beta_layer = nn.Linear(controller_output_size, 1)
        self.erase_layer = nn.Linear(controller_output_size, mem_vector_size)
        self.add_layer = nn.Linear(controller_output_size, mem_vector_size)

    def forward(self, controller_output, memory, prev_weights):
        key = torch.tanh(self.key_layer(controller_output))
        beta = F.softplus(self.beta_layer(controller_output))
        similarity = F.cosine_similarity(key.unsqueeze(1), memory, dim=2)
        write_weights = F.softmax(beta.unsqueeze(1) * similarity, dim=1)

        erase_vector = torch.sigmoid(self.erase_layer(controller_output))
        add_vector = torch.tanh(self.add_layer(controller_output))

        batch_size = memory.size(0)
        expanded_weights = write_weights.view(batch_size, self.mem_size, 1)
        expanded_erase = erase_vector.view(batch_size, 1, self.mem_vector_size)
        expanded_add = add_vector.view(batch_size, 1, self.mem_vector_size)

        erase_matrix = torch.bmm(expanded_weights, expanded_erase)
        add_matrix = torch.bmm(expanded_weights, expanded_add)

        updated_memory = memory * (1 - erase_matrix) + add_matrix
        return updated_memory, write_weights

# 4. Controller
class Controller(nn.Module):
    def __init__(self, input_size, hidden_size, num_read_heads, mem_vector_size, output_size):
        super(Controller, self).__init__()
        self.lstm = nn.LSTMCell(input_size + num_read_heads * mem_vector_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.fc_heads = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, prev_state, read_vectors_list):
        batch_size = x.size(0)
        flattened_read_vectors = torch.cat([rv.view(batch_size, -1) for rv in read_vectors_list], dim=1)

        controller_input = torch.cat([x, flattened_read_vectors], dim=1)
        h_t, c_t = self.lstm(controller_input, prev_state)

        model_output = self.fc_output(h_t)
        head_params = self.fc_heads(h_t)
        return model_output, head_params, (h_t, c_t)
