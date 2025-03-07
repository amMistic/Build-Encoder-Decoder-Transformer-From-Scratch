import math
import torch.nn as nn

class InputEmbedding(nn.Module):
    
    def __init__(self, d_model:int, vocab_size:int):
        '''
        Args:
            d_model : Embedding model dimension
            sequence_length : Total Length of input sequence string
        '''
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        '''
        Args:
            X : Input Sequence
        '''
        return self.embedding(x)  * math.sqrt(self.d_model)
        
        