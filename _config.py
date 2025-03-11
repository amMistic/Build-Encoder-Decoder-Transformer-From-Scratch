from pathlib import Path

def get_config():
    return {
        'batch_size' : 8,
        'num_epochs' : 2,
        'lr' : 10**-4,
        'sequence_length' : 350,
        'd_model' : 512,
        'src_lang' : 'en',
        'target_lang' : 'it',
        'model_folder' : "weights",
        'model_basename' : 'transformer_model',
        'preload' : None,
        'tokenizer_file' : 'tokenizer_{0}.json',
        "experiment_name" : 'runs/transformer_model'
    }
    
def get_weight_file_path(config, epoch : str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.')/ model_folder / model_filename)