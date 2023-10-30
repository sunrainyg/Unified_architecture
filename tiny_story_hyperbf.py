import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data.utils as tt_utils
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm.auto import tqdm
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)

dataset = load_dataset("roneneldan/TinyStories")
limit_train = 30000
limit_val = 100

data = dataset['train'][:limit_train]['text']
data_val = dataset['validation'][:limit_val]['text']

# Get the tokenizer with text normalization
tokenizer = tt_utils.get_tokenizer('basic_english')

# Normalize the stories
normalized_stories = [tokenizer(story) for story in data]

from collections import Counter

stories_dictionary = Counter()

for story in normalized_stories:
  stories_dictionary.update(set(story))

stories_counts = sorted([(x, stories_dictionary[x]) for x in stories_dictionary], key=lambda x: -x[-1])

word2index = {pair[0]:i for i, pair in enumerate(stories_counts, 1)}; word2index['<pad>'] = 0
index2word = {i:word[0] for i, word in enumerate(stories_counts, 1)}; index2word[0] = '<pad>'

def list2index(text_list):
  output = [word2index[word] for word in text_list]
  return output

def index2list(index_list):
  output = [index2word[index] for index in index_list]
  return output

def input_target_pair(input, maxlen=32):
  output = []
  for i in range(0, len(input)-(maxlen+1)):
    output.append((torch.tensor(input[i:i+maxlen]).to(device),
                   torch.tensor(input[i+maxlen]).to(device)))
  return output

# Function to process list of words to a dataloader of tuples (input, target)
def get_dataloader(stories, batch_size=16, maxlen=32):
  # Tokenize and pad lists of stories
  token_stories = [[0] * (maxlen-1) + list2index(story) for story in stories]

  # Convert tokenized stories to (input, target) tuples
  input_target_pairs = [input_target_pair(t_story, maxlen) for t_story in token_stories]

  # Store all examples in one list
  inpurt_target_heap = []

  for pair in input_target_pairs:
    inpurt_target_heap += pair

  # Generate dataloader
  dataloader = DataLoader(inpurt_target_heap, batch_size=batch_size, shuffle=True)
  return dataloader

def generate_square_subsequent_mask(sz):
  mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
  return mask


def create_mask(src):
  PAD_IDX = 0

  src_seq_len = src.shape[0]

  src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

  src_padding_mask = (src == PAD_IDX).transpose(0, 1)
  return src_mask, src_padding_mask

class TokenEmbedding(nn.Module):
  def __init__(self, d_model, vocab_size, dropout):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.dropout(self.embedding(x))

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, maxlen):
    super().__init__()

    den = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)).to(device)
    pos = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1).to(device)
    self.encoding = torch.zeros(maxlen, d_model).to(device)
    self.encoding[:, 0::2] = torch.sin(pos * den)
    self.encoding[:, 1::2] = torch.cos(pos * den)

  def forward(self, x):
    return x + self.encoding
  
class RBF_MultiHeadSelfAttention(nn.Module):
  def __init__(self, d_model, num_heads, dropout):
    super().__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads

    # Define separate linear transformations for query, key, and value for each head
    self.query = nn.Linear(d_model, d_model)
    self.key = nn.Linear(d_model, d_model)
    self.value = nn.Linear(d_model, d_model)
    self.M = nn.Parameter(torch.randn(self.head_dim, self.head_dim))

    # Output linear layer for each head
    self.out = nn.Linear(d_model, d_model)

    self.norm = nn.LayerNorm(d_model)

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    batch_size, seq_len, emb_dim = x.size()

    # Linear transformations for query, key, and value for each head
    Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2); Q = self.dropout(Q)
    K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2); K = self.dropout(K)
    V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2); V = self.dropout(V)

    # Calculate attention scores and attention weights for each head with scaling factor
    # scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5) # original
    
    ## Mahalanobis distance calculation
    # 1. Compute q^T M q
    q_M = Q @ self.M
    qMq = (Q * q_M).sum(dim=-1, keepdim=True)  # B, num_heads, N, 1
    # 2. Compute k^T M k
    k_M = K @ self.M
    kMk = (K * k_M).sum(dim=-1, keepdim=True).transpose(-2, -1)  # B, num_heads, 1, N
    # 3. Compute -2 q^T M k
    qMk = q_M @ K.transpose(-2, -1)  # B, num_heads, N, N
    negative_2qMk = -2 * qMk
    # Combine all components to get Mahalanobis distance
    dists = qMq + kMk + negative_2qMk
    scores = -dists * (self.head_dim ** 0.5)
    
    attention_weights = torch.softmax(scores, dim=-1)
    attended_values = torch.matmul(attention_weights, V)

    # Reshape and concatenate attended values from all heads
    attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    # Apply output linear layer and residual connection
    output = self.out(attended_values)
    output += x

    # Apply layer normalization
    output = self.norm(output)
    return output, attention_weights.detach()

#@title `FeedForward` class

class RBF_FeedForward(nn.Module):
    def __init__(self, d_model, center_feature, dropout=0.):
        super().__init__()

        self.beta_mean_history = []
        
        center_feature = int(50)
        
        # RBF Layer
        self.centers = nn.Parameter(torch.randn(center_feature, d_model))
        self.beta = nn.Parameter(torch.ones(center_feature) * 0.001)  # scale factor
        self.fc = nn.Linear(center_feature, d_model, bias=False) # set bias as false

        # Weights normalization
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')

    def radial_function(self, x):
        # Compute the distance from the centers
        A = x.pow(2).sum(dim=-1, keepdim=True)
        B = self.centers.pow(2).sum(dim=1)
        C = 2 * x @ self.centers.t()
        distances = A - C + B
        
        current_beta_mean = self.beta.mean().item()
        self.beta_mean_history.append(current_beta_mean)
        
        return torch.exp(-self.beta.unsqueeze(0) * distances)

    def forward(self, x):
        rbf_out = self.radial_function(x)
        rbf_out = self.fc(rbf_out)
        return rbf_out


class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout):
    super().__init__()

    # Linear layers
    self.pickles = nn.Linear(d_model, d_ff)
    self.tomatoes = nn.Linear(d_ff, d_model)

    self.norm = nn.LayerNorm(d_model)

    self.dropout = nn.Dropout(dropout)

    # Weights normalization
    nn.init.kaiming_normal_(self.pickles.weight, nonlinearity='relu')
    nn.init.kaiming_normal_(self.tomatoes.weight, nonlinearity='relu')

  def forward(self, x):
    pickle = self.pickles(x)
    pickle = F.relu(pickle)

    tomato = self.tomatoes(pickle)
    tomato = self.dropout(tomato)

    output = self.norm(tomato)
    return output

class MHDecoderTransformer(nn.Module):
  def __init__(self, d_model, maxlen, vocab_size, dropout, n_heads, d_ff, n_att):
    super().__init__()

    # Classes
    self.embedding = TokenEmbedding(d_model, vocab_size, dropout).to(device)
    self.posencoding = PositionalEncoding(d_model, maxlen).to(device)
    self.sequential_attention = [RBF_MultiHeadSelfAttention(d_model, n_heads, dropout).to(device) for _ in range(n_att)]
    self.neuralnet = RBF_FeedForward(d_model, d_ff, dropout).to(device)

    self.flatten = lambda x: x.view(x.size(0), -1)
    self.out = nn.Linear(maxlen * d_model, vocab_size)

  def forward(self, x):
    embeded = self.embedding(x)
    posencoded = self.posencoding(embeded)
    att_Ws = []

    attended = posencoded
    for lil_attention in self.sequential_attention:
      attended, att_W = lil_attention(attended)
      att_Ws.append(att_W)

    boring = self.neuralnet(attended)
    flat = self.flatten(boring)
    output = self.out(flat)
    return output, att_Ws

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, loader, plot_loss, print_loss):
  model.train()

  LOSS = 0
  plot = []

  for i, (input, target) in enumerate(loader):
    optimizer.zero_grad()

    logits, _ = model(input)

    loss = criterion(logits, target)
    LOSS += loss.item()
    loss.backward()

    optimizer.step()

    if (i+1)%print_loss == 0:
      print(f'Training epoch {i+1}/{len(loader)}: {(LOSS/i):.5f}')

    if i%plot_loss == 0:
      plot.append(loss.item())

  LOSS /= len(loader)
  return LOSS, plot

def get_time(epoch_time):
  minutes = int(epoch_time) // 60
  seconds = epoch_time - minutes*60
  return f'Time taken: {minutes} m. {seconds:.1f} s.'

def eval_model(model, loader, limit=1):
  model.eval()

  LOSS = 0
  loss_list = []

  with torch.no_grad():
    for i, (input, target) in enumerate(loader, 1):
      logits, _ = Optimus(input)

      loss = criterion(logits, target)

      LOSS += loss
      loss_list.append(loss)

      if (i / len(loader)) > limit:
        break

  LOSS /= len(loader) * limit
  return LOSS

def apply_temperature(logits, temperature=1.0):
    return logits / temperature

if __name__ == '__main__':
    BATCH_SIZE = 2048
    D_MODEL = 256
    VOCAB_SIZE = len(word2index)
    MAXLEN = 32
    NUM_HEADS = 8
    D_HIDDEN = 512
    N_ATT = 8
    DROPOUT = .1
    lr = 3e-4 * 5
    
    epochs = 20
    print_loss = 100
    plot_loss = 100
    loss_list = []

    Optimus = MHDecoderTransformer(D_MODEL, MAXLEN, VOCAB_SIZE, DROPOUT, NUM_HEADS, D_HIDDEN, N_ATT).to(device)

    num_params = count_parameters(Optimus)
    print(f"Number of trainable parameters in the model: {num_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=Optimus.parameters(), lr=lr)
    
    trainloader = get_dataloader(normalized_stories[:24000], BATCH_SIZE, MAXLEN)
    testloader = get_dataloader(normalized_stories[24000:], 32, MAXLEN)
    print("Data is successfully loaded!")

    # Training loop
    for epoch in tqdm(range(1, epochs+1)):
        start_time = time.time()
        loss, plot = train_epoch(Optimus, trainloader, plot_loss, print_loss)
        loss_list += plot
        epoch_time = time.time() - start_time
        print(f'Epoch #{epoch}: Loss = {loss:.5f}\n{get_time(epoch_time)}')
        validation_loss = eval_model(Optimus, testloader)
        print(f'Validation loss = {validation_loss:.5f}')
    
        # Save model's weights
        PATH = '/om2/group/cbmm/data/llm_hyperbf_epoch50_bs2048.pth'
        torch.save(Optimus.state_dict(), PATH)

    # inference
    input, target = next(iter(testloader))
    max_token = 250
    temperature = 1.

    current = input[0].unsqueeze(0)
    stack = input[0].tolist()
    
    # 使用index2list函数将输入的索引转换为单词列表
    input_sentence_tokens = index2list(input[0].tolist())

    # 将单词列表转换为一个字符串
    input_sentence = ' '.join(input_sentence_tokens)

    # 打印句子
    print("Input Sentence:", input_sentence)

    with torch.no_grad():
        for i in range(max_token):
            ex, _ = Optimus(current)

            # Apply temperature to the logits before sampling
            scaled_logits = apply_temperature(ex[0], temperature)
            probabilities = torch.softmax(scaled_logits, dim=-1)

            # Sample the next token using the probabilities distribution
            ex = torch.multinomial(probabilities, num_samples=1).squeeze()

            stack.append(ex.item())
            current = torch.tensor([stack[-32:]]).to(device)

    sentence = index2list(stack)

    # Create a single string with elements separated by a space
    sentence_string = ' '.join(sentence)

    # Set the number of elements to include in each line before inserting a newline character
    elements_per_line = 32

    # Split the sentence into chunks of 'elements_per_line' elements
    chunks = [sentence[i:i+elements_per_line] for i in range(0, len(sentence), elements_per_line)]

    # Join the chunks with a newline character to create line breaks
    formatted_sentence = '\n'.join(' '.join(chunk) for chunk in chunks)

    print("formatted_sentence:", formatted_sentence)
