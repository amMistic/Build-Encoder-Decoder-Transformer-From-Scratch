import torch.nn as nn
from .transformer import Transformer
from .decoder.decoder import Decoder
from .input_embeddding import InputEmbedding
from .projection_layer import ProjectionLayer
from .decoder.decoder_block import DecoderBlock
from .encoder.encoder import EncoderBlock, Encoder
from .positional_encoding import PositionalEncoding
from .encoder.feed_forward_layer import FeedForward
from .encoder.multi_headed_attention_block import MultiheadAttention

def build_transformer(
    src_vocab_size : int,
    target_vocab_size : int,
    src_sequence_length : int,
    target_sequence_length : int,
    d_model : int = 512,
    d_ff : int = 2048,
    N : int = 6,     # Number of encoder decoder block to connect with each other in the transformer
    h : int = 8,     # number of heads (multihead attention block)
    dropout : float= 0.1
    ) -> Transformer:
    
    # Create Embedding Layer
    src_embed = InputEmbedding(d_model, src_vocab_size)
    target_embed = InputEmbedding(d_model, target_vocab_size)
    
    # Create Positional Encoding Layer
    src_pos_encoding = PositionalEncoding(d_model, src_sequence_length, dropout)
    target_pos_encoding = PositionalEncoding(d_model, target_sequence_length, dropout)
    
    # Create Encoder Block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        feedforward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feedforward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create Decoder Block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttention(d_model, h, dropout)
        feedforward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feedforward_block, dropout)
        decoder_blocks.append(decoder_block)
      
    # Create Encoder - Decoder Block
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create Projection Layer
    projection_layer = ProjectionLayer(d_model, target_vocab_size)
    
    # Create Transfomer
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embedding=src_embed,
        target_embedding=target_embed,
        src_positional_encoding=src_pos_encoding,
        trg_positional_encoding=target_pos_encoding,
        projection_layer=projection_layer
    )
    
    # Initialized parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # 
    return transformer
        