import torch 
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model : int, sequence_length : int, dropout : float):
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)
        
        # create positional embedding matrix
        pos_embedding = torch.zeros(sequence_length, d_model)
        
        # positional encoding estimating terms
        pos = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        divisor_term = torch.exp(torch.arange(0, d_model, 2)).float() * (-math.log(10000.0) / d_model)
        
        # applying trigonometric functions
        pos_embedding[:, 0::2] = torch.sin(pos * divisor_term)
        pos_embedding[:, 1::2] = torch.cos(pos * divisor_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        
        # save the positional into the module
        self.register_buffer('pos_embedding', pos_embedding)
    
    def forward(self, x):
        x = x + (self.pos_embedding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)     
       