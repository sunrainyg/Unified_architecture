# pyt
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader

# data pipeline
from datasets import load_dataset, DatasetDict, load_from_disk
from typing import cast
import math, random

# tokenization
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

# logging
import os, argparse
t.set_default_device('mps')
hyper = {
    'vs': 2**13,
    'ly': 4,
    'hs': 768,
    'ah': 4,
    'cx': 512,
    'lr': 1e-4,
    'bs': 256,
    'ac': 4,
    'ep': 10,
}

hyper = argparse.Namespace(**hyper)
dataset = cast(DatasetDict, load_dataset('skeskinen/TinyStories-Instruct-hf'))
dataset['train'].set_format(type='torch', columns=['text'])
dataset['train'].format['type']
dataset['validation'].set_format(type='torch', columns=['text'])
dataset['validation'].format['type']
print(dataset)

tok = Tokenizer(BPE())
tok.normalizer = Lowercase()
tok.pre_tokenizer = ByteLevel()
tok.decoder = ByteLevelDecoder()
tok.post_processor = TemplateProcessing(single='$0 <|endoftext|>', special_tokens=[('<|endoftext|>', 1)],)
tok.enable_truncation(max_length=hyper.cx)
tok.enable_padding(pad_token='<pad>', length=hyper.cx)
trainer = BpeTrainer(vocab_size=hyper.vs, initial_alphabet=ByteLevel.alphabet(), special_tokens=['<pad>', '<|endoftext|>', '\n','Words: ', 'Features: ', 'Random sentence: ', 'Summary: ', 'Story: '])

if os.path.isfile('tiny.json'): tok = Tokenizer.from_file('tiny.json')
else:
  batch_size = 10000  # 可以根据您的需要调整
  total = len(dataset['train']['text'])
  for i in range(0, total, batch_size):
      batch = dataset['train']['text'][i:i+batch_size]
      tok.train_from_iterator(batch, trainer=trainer)
  tok.save('tiny.json')

  tok.train_from_iterator(dataset['train']['text'], trainer=trainer); tok.save('tiny.json')

tok = PreTrainedTokenizerFast(tokenizer_object=tok)
tok.pad_token = "[PAD]"

from tqdm import tqdm

def tokenization(example):
    return tok(example['text'], truncation=True, max_length=hyper.cx, padding='max_length')

if os.path.exists('/om/user/yulu_gan/train_dataset') and os.path.exists('/om/user/yulu_gan/valid_dataset'):
    train = load_from_disk('/om/user/yulu_gan/train_dataset')
    valid = load_from_disk('/om/user/yulu_gan/valid_dataset')

else:
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    tok.pad_token = '[PAD]'
    tok.pad_token_id = tok.convert_tokens_to_ids('[PAD]')

    train = dataset['train'].map(tokenization, batched=True, batch_size=8192, writer_batch_size=8192, load_from_disk=True)
    valid = dataset['validation'].map(tokenization, batched=True, batch_size=8192, writer_batch_size=8192, load_from_disk=True)
    
    train.save_to_disk('/om/user/yulu_gan/train_dataset')
    valid.save_to_disk('/om/user/yulu_gan/valid_dataset')

