import torch
from torch.utils.data import DataLoader, Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, token_ids, context_size, stride):
        self.input_ids = []
        self.target_ids = []

        for index in range(0, len(token_ids) - context_size + 1, stride):
            input_chunk = token_ids[index:index+context_size]
            target_chunk = token_ids[index+1:index+context_size+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return  self.input_ids[index], self.target_ids[index]


def create_data_loader(token_ids, batch_size=4, context_size=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = SlidingWindowDataset(token_ids=token_ids, context_size=context_size, stride=stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

