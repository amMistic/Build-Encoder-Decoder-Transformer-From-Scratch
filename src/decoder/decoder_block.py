import torch 
import torch.nn as nn

from src.encoder.multi_headed_attention_block import MultiheadAttention, ResidualConnection
from src.encoder.feed_forward_layer import FeedForward
from src.encoder.layer_normalization import LayerNormalization

class DecoderBlock(nn.Module):
    
    def __init__(self, 
                self_attention_block : MultiheadAttention, 
                cross_attention_block : MultiheadAttention, 
                feedforward_block : FeedForward, 
                dropout : float
    ):
        
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feedforward = feedforward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, trg_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, trg_mask))
        x = self.residual_connections[1](x, lambda x : self.cross_attention_block(x, encoder_output, encoder_output, trg_mask))
        x = self.residual_connections[2](x, self.feedforward)
        return x        