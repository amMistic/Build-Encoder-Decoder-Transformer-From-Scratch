from torch.utils.data import DataLoader, random_split

from pathlib import Path
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from _dataset import BilingualDataset, causal_mask
from src.build_transformer_ import build_transformer


def get_tokenizer(config, dataset, lang) -> Tokenizer:
    '''
    Construct the tokenizer for our desired language 
    
    Args:
        config : For Configuration
        dataset : Dataset on which our tokenizer will train on
        lang : Language for which you want tokenizer to tokenized the words or tokens
    
    Returns:
        Tokenizer
    '''
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer


def get_all_sentences(dataset, lang):
    '''
    Get all the sentences of one lang from the defined dataset
    
    Args:
        Dataset: Dataset of the sentences in our source and target langugae
        lang : Languages in which we want our sentences 
    
    Return:
        Sentence of our desired language only
    '''
    for item in dataset:
        yield item['translation'][lang]
  
        
def get_maximum_sequence_length(dataset, src_tokenizer : Tokenizer, target_tokenizer : Tokenizer, config, src_lang, target_lang):
    '''
    Calculate maximum source language sequence length and target language sequence length
    
    Args:
        dataset: Raw Dataset consiting pair of source language and target language
        src_tokenizer : Source Language Tokenizer
        target_tokenizer : Target Language Tokenizer
        config : Configuration
        src_lang : Source Language
        target_lang : Target Languaage
    
    Returns:
        Tuple of maximum sequence length of source language and target language
    
    '''
    max_seq_len_src = 0
    max_seq_len_target = 0
    
    for item in dataset:
        src_ids = src_tokenizer.encode(item['translation'][config['src_lang']]).ids
        target_ids = target_tokenizer.encode(item['translation'][config['target_lang']]).ids
        max_seq_len_src = max(max_seq_len_src, len(src_ids))
        max_seq_len_target = max(max_seq_len_target, len(target_ids))
    
    return max_seq_len_src, max_seq_len_target
    

def get_dataset(config):
    '''
    Get the dataset for our transformer
    
    Args:
        config : To configured our tokenizer
    '''
    dataset_raw = load_dataset('opus_books',f'{config['src_lang']}-{config['target_lang']}', split='train')
    
    # Build the tokenizer   
    src_tokenizer = get_tokenizer(config, dataset_raw, config['src_lang'])
    target_tokenizer = get_tokenizer(config, dataset_raw, config['target_lang']) 
    
    # Split the dataset into training and validation set
    training_size = int(0.9 * len(dataset_raw))
    validation_size = len(dataset_raw) - training_size
    
    training_dataset_raw, validation_dataset_raw = random_split(dataset_raw, [training_size, validation_size])
    trainig_dataset = BilingualDataset(training_dataset_raw, src_tokenizer, target_tokenizer, config['src_lang'], config['target_lang'], config['sequence_length'])
    validation_dataset = BilingualDataset(validation_dataset_raw, src_tokenizer, target_tokenizer, config['src_lang'], config['target_lang'], config['sequence_length'])
    
    # get the maximum sequence length
    max_seq_length_src, max_seq_length_target = get_maximum_sequence_length( dataset_raw, src_tokenizer, target_tokenizer, config, config['src_lang'], config['target_lang'])
    
    # Load the dataset into Module
    train_dataloader = DataLoader(trainig_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    
    # 
    return train_dataloader, validation_dataloader, src_tokenizer, target_tokenizer


def get_model(config, src_vocab_size, target_vocab_size):
    model = build_transformer(
        src_vocab_size = src_vocab_size,
        target_vocab_size=target_vocab_size,
        src_sequence_length=config['sequence_length'],
        target_sequence_length=config['sequence_length'],
        d_model=config['d_model']
    )
    return model