import torch
import argparse
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


import torch.nn as nn

class CustomMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()

        # Ensure the embedding size is divisible by number of heads
        assert embed_size % heads == 0
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # Combined key, query, and value projections in one matrix
        self.c_attn = nn.Linear(embed_size, 3 * embed_size)
        
        # Out projection
        self.c_proj = nn.Linear(embed_size, embed_size)
        
        self.attn_dropout = nn.Dropout(0.1)  # Default attention dropout used in GPT-2
        self.resid_dropout = nn.Dropout(0.1)  # Default residual dropout used in GPT-2

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.heads, x.size(-1) // self.heads)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)  

    def merge_heads(self, x):
        return x.contiguous().view(x.size(0), x.size(2), x.size(1) * x.size(3))

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.embed_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        
        a = torch.matmul(query, key.transpose(-2, -1))
        a = a / (self.head_dim ** 0.5)
        a = torch.nn.functional.softmax(a, dim=-1)
        a = self.attn_dropout(a)
        
        y = torch.matmul(a, value)
        y = self.merge_heads(y)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        
        return y

class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config):
        super().__init__(config)
        # Replace the standard multihead attention with your custom implementation
        self.attn = CustomMultiHeadSelfAttention(
            embed_size=config.n_embd,
            heads=config.n_head
        )
        
class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Replace the attention module in each block with the custom attention
        for i in range(config.n_layer):
            self.transformer.h[i].attn = CustomGPT2Attention(config)




class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def generate_story(prompt, model, tokenizer, max_length=500):

    device          = model.device
    input_ids       = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output          = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id, 
                            no_repeat_ngram_size=2, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)

    generated_text  = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return prompt + generated_text


def main():
    parser              = argparse.ArgumentParser(description='Fine-tune GPT-2/HyperBF on custom dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', metavar='PATH', type=str, required=True,
                        help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
    parser.add_argument('--epoches', type=int, default=20, help='Epoches')
    parser.add_argument('--model_path', type=str, default='/om/user/yulu_gan/model/gpt2_transformer.pth', help='Path to the pretrained model')
    parser.add_argument('--resume', type=bool, default=False, help='If resume from pretrained model; Set this as True when doing testing')
    
    args                    = parser.parse_args()
    print(args)
    
    if args.dataset         == 'tiny_story':
        
        dataset             = load_dataset("roneneldan/TinyStories")
        limit_train         = 30000
        limit_val           = 100

    data                    = dataset['train'][:limit_train]['text']
    data_val                = dataset['validation'][:limit_val]['text']

    # Train
    if not args.resume:
        tokenizer           = GPT2Tokenizer.from_pretrained('gpt2-medium')
        tokenizer.pad_token = tokenizer.eos_token
        
        config              = GPT2Config(vocab_size=tokenizer.vocab_size, n_layer=1)
        model               = CustomGPT2LMHeadModel(config)

        train_encodings     = tokenizer(data, truncation=True, padding=True, return_tensors='pt')
        val_encodings       = tokenizer(data_val, truncation=True, padding=True, return_tensors='pt')

        train_dataset       = TinyStoriesDataset(train_encodings)
        val_dataset         = TinyStoriesDataset(val_encodings)

        data_collator       = DataCollatorForLanguageModeling(
                                tokenizer=tokenizer,
                                mlm=False
                            )

        training_args       = TrainingArguments(
                                output_dir="./results",
                                overwrite_output_dir=True,
                                num_train_epochs=1,
                                per_device_train_batch_size=8,
                                per_device_eval_batch_size=8,
                                eval_steps=400,
                                save_steps=800,
                                warmup_steps=500,
                                logging_dir='./logs',
                            )

        trainer             = Trainer(
                                model=model,
                                args=training_args,
                                data_collator=data_collator,
                                train_dataset=train_dataset,
                                eval_dataset=val_dataset,
                                compute_metrics=None,
                            )

        for epoch in range(args.epoches):
            print("we are training at the epoch{}".format(epoch))
            trainer.train()
            if epoch % 10   == 0:
                model.save_pretrained(args.model_path)

    # Test
    else:
        model               = GPT2LMHeadModel.from_pretrained(args.model_path)
        tokenizer           = GPT2Tokenizer.from_pretrained('gpt2-medium')
        
        device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model               = model.to(device)
        prompt              = "Once upon a time, in a land far away,"
        generated_story     = generate_story(prompt, model, tokenizer)
        print(generated_story)


if __name__ == '__main__':
    main()

