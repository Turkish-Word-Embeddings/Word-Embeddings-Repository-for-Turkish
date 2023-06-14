import torch

class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        # return length of dataset
        return len(self.X)
    
    def __getitem__(self, index):
        # given an index (item), return review and target of that index in torch tensor
        sentence = torch.tensor(self.X[index,:], dtype = torch.long)
        tags = torch.tensor(self.y[index], dtype = torch.long)
        
        return {'sentence': sentence,
                'tags': tags}