import torch 
import torch.nn as nn

class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model:int, vocab_size : int):
        super().__init__()
        self.projection_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # map the each result token to its corresponding word or character in the vocabulary
        # dimension (batch_size, sequence_length, d_model) -->> (batch_size, sequence_length, vocab_size)
        return torch.log_softmax(self.projection_layer(x), dim = -1)