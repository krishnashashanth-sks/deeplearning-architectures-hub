import torch

#  Data Generation Function
def generate_copy_sequence(batch_size, min_seq_len, max_seq_len, vector_size, start_delimiter, end_delimiter):
    seq_len = torch.randint(min_seq_len, max_seq_len + 1, (1,)).item()
    data_vector_size = vector_size - 2
    random_data = torch.randint(0, 2, (batch_size, seq_len, data_vector_size)).float()

    zeros_for_data_prefix = torch.zeros(batch_size, seq_len, 2).float()
    padded_random_data = torch.cat([zeros_for_data_prefix, random_data], dim=2)

    batch_start_delimiter = start_delimiter.expand(batch_size, -1).unsqueeze(1)
    batch_end_delimiter = end_delimiter.expand(batch_size, -1).unsqueeze(1)

    input_sequence = torch.cat([
        batch_start_delimiter,
        padded_random_data,
        batch_end_delimiter
    ], dim=1)

    input_padding_length = seq_len + 2
    target_padding = torch.zeros(batch_size, input_padding_length, vector_size).float()

    copied_data_for_target = torch.cat([
        torch.zeros(batch_size, seq_len, 2).float(),
        random_data
    ], dim=2)

    target_sequence = torch.cat([
        target_padding,
        copied_data_for_target
    ], dim=1)

    return input_sequence, target_sequence
