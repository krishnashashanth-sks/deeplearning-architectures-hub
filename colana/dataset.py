from torch.utils.data import Dataset
import torch

# ---  CoNaLaDataset Class ---

class CoNaLaDataset(Dataset):
    def __init__(self, nl_ids, code_ids, pad_idx, max_nl_len=None, max_code_len=None):
        self.nl_ids = nl_ids
        self.code_ids = code_ids
        self.pad_idx = pad_idx

        if max_nl_len is None:
            self.max_nl_len = max([len(seq) for seq in nl_ids])
        else:
            self.max_nl_len = max_nl_len

        if max_code_len is None:
            self.max_code_len = max([len(seq) for seq in code_ids])
        else:
            self.max_code_len = max_code_len

        self.padded_nl_ids = [self._pad_sequence(seq, self.max_nl_len) for seq in nl_ids]
        self.padded_code_ids = [self._pad_sequence(seq, self.max_code_len) for seq in code_ids]

    def _pad_sequence(self, sequence, max_len):
        if len(sequence) < max_len:
            return sequence + [self.pad_idx] * (max_len - len(sequence))
        return sequence[:max_len]

    def __len__(self):
        return len(self.nl_ids)

    def __getitem__(self, idx):
        return {
            'nl_input': torch.tensor(self.padded_nl_ids[idx], dtype=torch.long),
            'code_target': torch.tensor(self.padded_code_ids[idx], dtype=torch.long)
        }

