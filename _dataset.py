import torch 
from torch.utils.data import Dataset
from tokenizers import Tokenizer

# Global Method
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0            # reason behind is torch.triu makes all the values in upper triangle as 1 and 
                                # lower triangle values as 0
                                # by making mask == 0 ----> convert 0 -> True (1) where cell value is 0 and False(0) where it isn't.
                                # this gives us our desired matrix
                                
                                 
class BilingualDataset(Dataset):
    
    def __init__(self, 
                dataset,
                src_tokenizer : Tokenizer,
                target_tokenizer : Tokenizer,
                src_lang,
                target_lang,
                sequence_length)-> None:
        
        super().__init__()
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.sequence_length = sequence_length
        
        self.sos_token = torch.tensor([target_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([target_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([target_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # get the src-target lang sentence pair from the dataset
        src_target_sentence = self.dataset[index]
        src_text = src_target_sentence['translation'][self.src_lang]
        target_text = src_target_sentence['translation'][self.target_lang]        
        
        # apply tokenizer on both source and target language
        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        dec_input_tokens = self.target_tokenizer.encode(target_text).ids
        
        # apply padding on the tokenized tokens 
        enc_token_padding = self.sequence_length - len(enc_input_tokens) - 2        # substracting 2 for SOS and EOS
        dec_token_padding = self.sequence_length - len(dec_input_tokens) - 1       # substracting 1 because while decoding we are only using <SOS>
        
        # validating the padding and input texts
        if enc_token_padding < 0 or dec_token_padding < 0:
            raise ValueError('Sentence is too long.')

        # Create Encode Input by adding SOS and EOS token to the input texts
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_token_padding, dtype=torch.int64)
            ],
            dim = 0
        ) 
        
        # Create Decoder Input by adding SOS tokens
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_token_padding, dtype=torch.int64)
            ],
            dim=0 
            
        )
        
        # Create Lable / Target Sentence in our target language by adding EOS token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_token_padding, dtype=torch.int64)
            ],
            dim = 0
        )
        
        # Valdate whether each tensors were of same length or not
        assert encoder_input.size(0) == self.sequence_length
        assert decoder_input.size(0) == self.sequence_length
        assert label.size(0) == self.sequence_length 

        #
        return {
            'encoder_input' : encoder_input,
            'decoder_input' : decoder_input,
            
            # Encoder mask ----> masked the padding tokens 
            'encoder_mask' : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),          # dimension(1, 1, sequence_length)
            
            # Decoder Mask ----> masked the padding tokens and all the values which are above the diagonal in Multihead attention matrix
            'decoder_mask' : (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            
            'label' : label,
            'src_text' : src_text,
            'target_text' : target_text
        }
        
        