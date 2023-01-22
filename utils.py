import os
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
import torch, torch.nn as nn
from torch.nn import functional as F

# hyperparameters
BATCH_SIZE = 64 # how many independent sequences will we process in parallel?
BLOCK_SIZE = 64 # what is the maximum context length for predictions?
MAX_ITER = 5000
EVAL_INTER = 500
LEARNNG_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
NUM_EMBED = 768 # 128*num_head
NUM_HEAD = 6
NUM_LAYER = 6
DROPOUT = 0.2


def encode(vocab:torchtext.vocab, text_seq: str)->list[int]:
    """
    Function to encode imput text using simple tokenizer
    """
    tokenizer = get_tokenizer("basic_english")
    return [vocab.get_stoi()[token] for token in tokenizer(text_seq)]

def decode(vocab:torchtext.vocab, enc_sec: list[int])->str:
    return " ".join(vocab.get_itos()[i] for i in enc_sec)

def build_vocab(path_to_data:str='data/english.txt', specials:list=['<unk>','<pad>','<bos>','<eos>'])->(torchtext.vocab, int):
    """
    The function build vocabulary from the input text dataset
    Parameters:
    path_to_data (str): path to the data file
    specials (list): list of special symbols
    Returns:
    vocab, vocab_size (vocab, int): vocabulary and its length
    """
    tokenizer = get_tokenizer("basic_english")
    iterator = _yield_tokens(path_to_data, tokenizer)
    vocab = build_vocab_from_iterator(iterator, specials=specials)
    vocab_size = len(vocab)
    return vocab, vocab_size

def _yield_tokens(filepath, tokenizer):
    '''
    Iterator through the tokens of the input
    We need iterator to build vocabulary with
    build_vocab_from_iterator from torchtext.vocab
    '''
    data_arr = open(filepath, encoding='utf-8').read().split('\n')
    for phrase in data_arr:
        tokens = tokenizer(phrase)
        yield tokens

def get_batch(data:list[str], block_size:int, batch_size:int):
    """
    This is a simple function to create batches of data. 
    GPUs allow for parallel processing we can feed multiple chunks at once
    so that's why we would need batches - how many independant sequences
    will we process in parallel.
    
    Parameters:
    data: list[str]: data to take batch from
    block_size (int): size of the text that is proccessed at once
    batch_size (int): number of sequences to process in parallel
        
    Returns:
    x, y: a tuple with token sequence and token target
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # we stack batch_size rows of sentences
    # so x and y are the matrices with rows_num=batch_size
    # and col_num=block_size
    x = torch.stack([data[i:i+block_size] for i in ix])
    # y is x shifted one position right - because we predict
    # word in y having all the previous words as context
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def load_model_from_checkpoint(model_class:torch.nn.Module, path_to_checkpoint:str='checkpoints/state_dict_model.pt', **kwargs:dict)->torch.nn.Module:
    try:
        state_dict = torch.load(path_to_checkpoint)
        print("Successfully loaded model from the checkpoint")
    except Exception as e:
        print(f"Error loading the model from the checkpoint. {e}")
    
    model = model_class(**kwargs)
    # load the state_dict into the model
    model.load_state_dict(state_dict)
    return model


def save_model_to_chekpoint(model:torch.nn.Module, path_to_checkpoint:str='checkpoints', suffix:str=''):
    checkpoint_name = 'state_dict_model'+suffix+'.pt'
    full_path = os.path.join(path_to_checkpoint, checkpoint_name)
    try:
        torch.save(model.state_dict(), path_to_checkpoint)
        print("Successfully saved the model to {}".format(full_path))
    except Exception as e:
        print(f"Error saving the model to checkpoint. {e}")