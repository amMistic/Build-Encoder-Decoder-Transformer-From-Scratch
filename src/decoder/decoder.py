import torch 
import torch.nn as nn

from .decoder_block import DecoderBlock
from src.encoder.layer_normalization import LayerNormalization

class Decoder(nn.Module):
    
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask,  targ_mask):
        for layers in self.layers:
            x = layers(x, encoder_output, src_mask, targ_mask)
        return self.norm(x)