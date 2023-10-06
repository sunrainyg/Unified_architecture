import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 1. 加载数据集
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories")
limit_train = 30000
limit_val = 100

data = dataset['train'][:limit_train]['text']
data_val = dataset['validation'][:limit_val]['text']

# 2. 初始化tokenizer和模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token

# model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

config = GPT2Config(vocab_size=tokenizer.vocab_size, n_layer=1)
model = GPT2LMHeadModel(config)

# 3. Tokenize数据集
train_encodings = tokenizer(data, truncation=True, padding=True, return_tensors='pt')
val_encodings = tokenizer(data_val, truncation=True, padding=True, return_tensors='pt')

# 4. 使用PyTorch Dataset类定义数据集
class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = TinyStoriesDataset(train_encodings)
val_dataset = TinyStoriesDataset(val_encodings)

# 5. 定义collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 6. 定义TrainingArguments和Trainer，然后训练模型
training_args = TrainingArguments(
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

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

epoches = 20
for epoch in range(epoches):
    print("we are training at the epoch{}".format(epoch))
    trainer.train()

# 在完成训练后，可以保存模型并在之后使用它
model.save_pretrained('/om2/group/cbmm/data/gpt2_transformer.pth')
# tokenizer.save_pretrained('./trained_model')

def generate_story(prompt, model, tokenizer, max_length=500):
    """
    根据给定的提示生成故事。
    
    参数：
        prompt (str): 提示或开始的文本
        model: 已经训练过的模型
        tokenizer: 对应的tokenizer
        max_length (int): 生成的文本的最大长度
    
    返回：
        str: 生成的文本
    """
    device = model.device

    # 使用tokenizer编码提示并转换为PyTorch tensors
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # 使用模型生成文本
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id, 
                            no_repeat_ngram_size=2, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)

    # 解码生成的文本
    generated_text = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return prompt + generated_text

# 示例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
prompt = "Once upon a time, in a land far away,"
generated_story = generate_story(prompt, model, tokenizer)
print(generated_story)

