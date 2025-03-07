import torch 
import torch.nn as nn

class FeedForward(nn.Module):
    
    def __init__(self, d_model: int, d_ff : int, dropout : float):
        '''
        Convert the input sentence into d_ff dimension using linear layer, applying Relu activation on the input sequence
        convert back into d_model dimensions. 
        Dropout is also applied in between these two linear layers to avoid overfitting
        
        Args:
            d_model : Emedding model Dimesions
            d_ff : FeedForward Model Dimensions
        '''
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and b1
        self.dropout = nn.Dropout(dropout)      # Dropout 
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        # input sequences are in batchs (Batch_size, sequence_length, d_model) 
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        