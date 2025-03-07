import math
import torch.nn as nn

from .layer_normalization import LayerNormalization

class MultiheadAttention(nn.Module):
    
    def __init__(self, d_model : int, h :int, dropout : float ):
        super().__init__()
        self.d_model = d_model
        self.h = h      # NUMBER OF HEADS
        self.dropout = nn.Dropout(dropout)      # add dropout layer to avoid overfitting the model
        
        # get the heads dimension
        assert d_model % h == 0
        self.dk = d_model // h
        
        # initialized parameter matrix ( Wq, Wk, Wv, Wo)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
    
    @staticmethod
    def cal_attention_scores(self, key_prime, query_prime, value_prime, mask, dropout : nn.Dropout):
        '''
        Args:
            Key_prime   : Resultant matrix after mutliplying the key matrix sith its respective parameters matrix
            query_prime : Resultant matrix after mutliplying the query matrix sith its respective parameters matrix
            value_prime : Resultant matrix after mutliplying the value matrix sith its respective parameters matrix
            mask        : Used to mask the token in the model while training to avoid interaction between modela and this
                        particular set of words/tokens
            dropout     : Applying Dropout layer to avoid overfitting
        
        Returns:
            attention_scores: Attention Scores
            X : resultant matrix after multiplying the attention scores with the values  
                   
        '''
        
        dk = key_prime.shape[-1]
        
        # dimesnion are (batch_size, h, sequence_length, dk) -- > (batch_size, h, sequence_length, sequence_length)
        attention_scores = (query_prime @ key_prime.transpose(-2,-1)) / math.sqrt(dk)
        
        # now, masked the tokens which are defined to masked 
        if mask is not None:
            attention_scores.masked_fill(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        
        # apply the dropout layer to resultant attention scores to avoid overfitting 
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value_prime), attention_scores
        
    def forward(self, k, q, v, mask):
        query_prime = self.Wq(q)      # Matrix dimensions (batch_size, sequence_length, d_model) --> (batch_size, sequence_length, d_model)
        key_prime = self.Wk(k)
        value_prime = self.Wv(v)
        
        # Create 'h' number of heads for each key, query, and value matrix
        # dimensions were (batch_size, sequence_length, d_model) -- > (batch_size, sequence_length, h, dk) ---> (batch_size, h, sequence_length, dk)
        query_heads = query_prime.view(query_prime.shape[0], query_prime.shape[1], self.h, self.dk).transpose(1, 2)
        key_heads = key_prime.view(key_prime.shape[0], key_prime.shape[1], self.h, self.dk).transpose(1, 2)
        value_heads = value_prime.view(value_prime.shape[0], value_prime.shape[1], self.h, self.dk).transpose(1, 2)
        
        # Calculate Attention scores
        x, self.attention_scores = MultiheadAttention.cal_attention_scores(key_heads, query_heads, value_heads, mask, self.dropout)
        
        # concatinate all the heads into one matrix
        # dimesnion were (batch_size, h, sequence_length, dk) --> (batch_size, sequence_length, h, dk) ---> (batch_size, sequence_legth, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.dk)

        # Get the final Multihead attention matrix
        return self.Wo(x)
        
        
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout : float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm_layer = LayerNormalization()
    
    def froward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm_layer(x)))
    
    