import torch.nn as nn
from .feed_forward_layer import FeedForward
from .layer_normalization import LayerNormalization
from .multi_headed_attention_block import MultiheadAttention, ResidualConnection

class EncoderBlock(nn.Module):
    '''
    Building Encoder Block by connecting two Add & Norm Block, One FeedForward Block and One Multihead 
    Attention Block
    
    Input Embedding --->> Attention result + Add & Norm --->> Feed Forward result + Add Norm --->>
    Decoder / Next Encoder 
    
    '''
    
    def __init__(self, 
                self_attention_block : MultiheadAttention ,
                feedforward_block : FeedForward ,
                dropout : float
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention_block = self_attention_block
        self.feedforward_block = feedforward_block
        
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        '''
        Args:
            X        : Input Embedding 
            src_mask : To mask the words or token to not interact with the model 
        '''
        # pass the input sequence from the Multihead attention block --> Add & Norm Block
        x = self.residual_connection[0](x, lambda x :self.attention_block(x, x, x, src_mask))
        
        # pass the updated sequence from the last layers/ block to feedforward + Add & Norm block 
        return self.residual_connection[1](x, self.feedforward_block)
        

class Encoder(nn.Module):
    
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm_layer = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm_layer(x)