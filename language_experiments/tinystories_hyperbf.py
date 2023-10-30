# # Copyright (c) 2023, Yulu Gan
# # Licensed under the BSD 3-clause license (see LICENSE.txt)
# # ---------------------------------------------------------------------------------
# # ** Description ** Training HyperBF on Tinystories datasets.
# # --------------------------------------------------------
# from datasets import load_dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm

# print("1")

# def generate_story(model, prompt, max_length=100, temperature=1.0):
#     model.eval()
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
#     generated_ids = input_ids
    
#     with torch.no_grad():
#         for _ in range(max_length - len(input_ids[0])):
#             logits = model(generated_ids)
#             next_token_logits = logits[:, -1, :] / temperature
#             next_token_id = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
#             generated_ids = torch.cat((generated_ids, next_token_id), dim=-1)

#             # Stop generation if EOS token is generated (assuming tokenizer.eos_token_id exists)
#             if next_token_id[0, 0].item() == tokenizer.eos_token_id:
#                 break
                
#     generated_text = tokenizer.decode(generated_ids[0])
#     return generated_text


# # Load dataset
# dataset             = load_dataset("roneneldan/TinyStories")
# # %%
# # Load pretrained model for experimentation

# model_name          = 'roneneldan/TinyStories-1M'
# model_download_path = '/om/user/yulu_gan/model'

# tokenizer           = AutoTokenizer.from_pretrained(model_name, cache_dir = model_download_path)
# pretrained_model    = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = model_download_path)

# # Experiment with the pretrained model
# # Generate completions
# prompt              = "The cat sat on the mat"
# input_ids           = tokenizer.encode(prompt, return_tensors="pt")
# greedy_output       = pretrained_model.generate(input_ids, max_length=200)
# print("Output:\n" + 100 * '-')
# print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
# # %%
# # Define a transformer to run on the dataset

# class Transformer(nn.Module):
#     def __init__(self, n_embd=512, vocab_size=50257):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, n_embd)
#         transformer_layer = nn.TransformerEncoderLayer(n_embd, 8)
#         self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=8, mask_check=True)
#         self.unembed = nn.Linear(n_embd, vocab_size)

#     def forward(self, x):
#         x = self.embed(x)
#         mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
#         x = self.transformer(x, mask=mask, is_causal=True)
#         x = self.unembed(x)
#         return x 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Transformer().to(device)
# # model(input_ids)

# if torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model)

# optim = torch.optim.Adam(model.parameters(), lr=1e-3)
# # %%
# portion = 0.010  # Use 10% of the data
# num_samples = int(len(dataset['train']) * portion)
# sub_dataset = torch.utils.data.Subset(dataset['train'], indices=range(num_samples))
# train_loader = DataLoader(sub_dataset, batch_size=64, shuffle=True)

# # %%

# tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# for epoch in range(1):
#     for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Batches")):
#         optim.zero_grad()

#         tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt')['input_ids'].to(device)
#         logits = model(tokenized)
#         # flatten out seq dim
#         loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), tokenized.view(-1))
#         tqdm.write(f"Loss: {loss.item()}")

#         loss.backward()
#         optim.step()
#         if batch_idx % 100 == 0:
#             torch.save(model.state_dict(), '/om/user/yulu_gan/model/1021_model_weights_path.pt')

# #Test
# prompt = "Once upon a time,"  # You can replace this with your desired starting text
# generated_story = generate_story(model, prompt, max_length=200)
# print(generated_story)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import random
from tqdm import tqdm
import os

# 定义模型

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        token_ids = tokenizer.encode(self.texts[idx], add_special_tokens=True, return_tensors='pt')
        return token_ids[0]  # 获取张量中的第一个元素，因为encode返回一个1D张量列表

    def __len__(self):
        return len(self.texts)


class TransformerGPT2(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerGPT2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(5000, d_model)  # 假设序列长度不超过5000

        # 使用TransformerEncoder代替完整的Transformer
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        batch_size, seq_length = src.size()
        positions = torch.arange(seq_length, device=src.device).unsqueeze(0).repeat(batch_size, 1)
        x = self.embedding(src) + self.position_embedding(positions)

        # 仅使用encoder
        x = self.transformer(x)

        x = self.fc(x)
        return x

# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = TransformerGPT2(vocab_size=50257, d_model=512, nhead=4, num_layers=1, dim_feedforward=512).to(device)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    
# 准备数据
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
dataset = load_dataset("roneneldan/TinyStories")

# 先获取前1000个样本
sampled_data = dataset['train'][:100000]
texts = sampled_data['text']

train_dataset = TinyStoriesDataset(texts, tokenizer)
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    return pad_sequence(batch, padding_value=0.0, batch_first=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 训练模型

if True:
    save_dir = "/om/user/yulu_gan/model/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(5):
        total_loss = 0
        num_batches = len(train_loader)
        
        # 使用tqdm包装train_loader
        train_loader_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{5}", unit="batch")
        
        for batch in train_loader_progress:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch.view(-1))
            loss.backward()
            optimizer.step()

            # 更新总损失
            total_loss += loss.item()
            
            # 使用set_postfix来显示平均损失
            train_loader_progress.set_postfix({"Avg. Loss": total_loss / (train_loader_progress.n + 1)}, refresh=True)

        print(f"Epoch {epoch+1} Loss: {total_loss/num_batches}")

        save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model for epoch {epoch+1} at {save_path}")


def generate_text(model, tokenizer, initial_text, max_length=50, top_k=50, top_p=0.95, temperature=0.7, no_repeat_ngram_size=2):
    generated = torch.tensor(tokenizer.encode(initial_text)).unsqueeze(0).to(device)
    past_tokens = set()
    
    with torch.no_grad():
        for _ in range(max_length - len(initial_text)):
            logits = model(generated)

            logits = logits[:, -1, :] / temperature  # 取最后一个时间步
            
            # 将非top_k的token的logits设置为负无穷
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            removed_tokens = sorted_indices[:, top_k:]
            logits[:, removed_tokens] = -float('Inf')
            
            # 使用top-p (nucleus) sampling
            sorted_logits /= temperature
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            removed_tokens = cumulative_probs > top_p
            sorted_indices_to_remove = sorted_indices[removed_tokens]
            logits[:, sorted_indices_to_remove] = -float('Inf')

            # 避免重复的n-grams
            for idx in range(logits.shape[1]):
                token = torch.tensor(idx).unsqueeze(0).unsqueeze(0).to(device)
                ngram = torch.cat((generated, token), dim=1)
                ngram_list = list(ngram[0, -no_repeat_ngram_size:].cpu().numpy())
                ngram_tuple = tuple(ngram_list)
                if ngram_tuple in past_tokens:
                    logits[0, idx] = -float('Inf')

            # 使用多项式分布采样下一个token
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("There are NaN or Inf values in logits!")

            clamped_logits = torch.clamp(logits, min=-10, max=10)
            probabilities = F.softmax(clamped_logits, dim=-1)

            if torch.isnan(probabilities).any() or torch.isinf(probabilities).any() or (probabilities < 0).any():
                print("There are invalid values in probabilities!")

            next_token = torch.multinomial(probabilities, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)
            
            # 更新past_tokens集合
            for start_idx in range(generated.shape[1] - no_repeat_ngram_size + 1):
                ngram_tuple = tuple(generated[0, start_idx:start_idx + no_repeat_ngram_size].cpu().numpy())
                past_tokens.add(ngram_tuple)
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)


if True:
    save_path           = '/om/user/yulu_gan/model/model_epoch_2.pth'
    model.load_state_dict(torch.load(save_path))
    tokenizer           = GPT2Tokenizer.from_pretrained('gpt2-medium')
    device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model               = model.to(device)
    prompt              = "Once upon a time, in a land far away,"
    generated_story     = generate_text(model, tokenizer, prompt)
    print(generated_story)
