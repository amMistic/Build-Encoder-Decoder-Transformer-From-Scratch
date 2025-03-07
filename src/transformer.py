import torch.nn as nn

from .positional_encoding import PositionalEncoding
from .projection_layer import ProjectionLayer
from .input_embeddding import InputEmbedding
from .encoder.encoder import Encoder
from .decoder.decoder import Decoder


class Transformer(nn.Module):
    
    def __init__(self, 
                encoder : Encoder, 
                decoder : Decoder, 
                src_embedding : InputEmbedding,
                target_embedding : InputEmbedding,
                src_positional_encoding : PositionalEncoding,
                trg_positional_encoding : PositionalEncoding,
                projection_layer : ProjectionLayer
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.src_pos_encoding = src_positional_encoding
        self.target_pos_encoding = trg_positional_encoding
        self.projection_layer = projection_layer
        
    
    # Embedd the encoder block into transformer model
    def encode(self, src, src_mask):
        # apply embedding model on input sequence
        src = self.src_embedding(src)
        
        # pass it throught positional encoding to get the positional embedding
        src = self.src_pos_encoding(src)
        
        # pass the updated input sequence through the encoder block
        return self.encoder(src, src_mask)

    
    # Embedd the decoder block into the transformer model
    def decode(self, trg, encoder_output, src_mask, trg_mask):
        # apply embedding model on target sequence
        trg = self.target_embedding(trg)
        
        # pass it through positional encoding to get the positional embedding
        trg = self.target_pos_encoding(trg)
        
        # pass the updated target_sequence through the decoder block
        return self.decoder(trg, encoder_output, src_mask, trg_mask)

    
    # Last step is to project the final output to map each token to its respective word or character in 
    # the vocabulary
    def projection_layer(self, x):
        return self.projection_layer(x)
        