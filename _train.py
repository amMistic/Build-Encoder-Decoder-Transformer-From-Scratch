import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from _utils import get_dataset, get_model
from _config import get_weight_file_path, get_config

from pathlib import Path
from tqdm import tqdm
import warnings


def train_model(config):
    # defined the device on which model to train on CUDA or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device : {device}')
    
    # Create model folder (if not)
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    # load the training and validation data loader
    train_dataloader, validation_dataloader, src_tokenizer, target_tokenizer = get_dataset(config)
    
    # load the model
    model = get_model(config, src_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)
    
    # load the tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    # define the optimzer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], eps=1e-9)
    
    # Save the state in case of model crash in between the training
    initial_epoch = 0
    global_step = 0         # use for tensorboard to keep track of the loss
    if config['preload']:
        # load the model again
        model_filename = get_weight_file_path(config, config['preload'])     # reason we're passing config['preload'] as epoch, is because we want model to start from where it stopped
        print('Model loaded sucessfully!')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    
    # define loss function to calculate the loss
    loss_fnc = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    # Training Loop
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        
        batch_iterator = tqdm(train_dataloader, desc=f'Processing Epoch : {epoch:02d}')
        for batch in batch_iterator:
            
            encoder_input = batch['encoder_input'].to(device)       # dimensions : (batch_size, sequence_length)
            decoder_input = batch['decoder_input'].to(device)       # dimensions : (batch_size, sequence_length)
            encoder_mask = batch['encoder_mask'].to(device)         # dimensions : (batch_size, 1, 1, sequence_length)
            decoder_mask = batch['decoder_mask'].to(device)        # dimensions : (batch_size, 1, sequence_length, sequence_length)
            
            # pass these data through transformer
            encoder_output = model.encode(encoder_input, encoder_mask)          # dimensions : (batch_size, sequence_length, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)        # dimensions : (batch_size, sequence_length, sequence_length)
            projection_output = model.projection_layer(decoder_output)      # dimensions : (batch_size, sequence_length, target_vocab_size)

            # get the actual label / target value
            label = batch['label'].to(device)           # dimensions  : (batch_size, sequence_length)
            
            # Compare the error
            # dimensions :  (batch_size, sequence_length, target_vocab_size) ---> ( batch_size * sequence_length, target_vocab_size)
            loss = loss_fnc(projection_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))
            
            # display the loss on the progress bar
            batch_iterator.set_postfix({'loss' : f'{loss.item() :6.3f}'})
            
            # log the loss on tensorboard
            writer.add_scalar('train_loss' , loss.item())
            writer.flush()
            
            # backpropagate the loss
            loss.backward()
            
            # update the weights
            optimizer.step()
            optimizer.zero_grad()
            
            # updat the global step
            global_step += 1
        
        # save current state of model
        model_filename = get_weight_file_path(config, epoch)
        torch.save(
            {
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'global_step' : global_step
            },
            model_filename
        )

if __name__ == '__main__' : 
    # ignore cuda warnings
    warnings.filterwarnings('ignore')
    
    # get the configuration file
    config = get_config()
    
    # train model
    train_model(config)