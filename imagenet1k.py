from datasets import load_dataset
print("1")
dataset = load_dataset("imagenet-1k", cache_dir="/nobackup/scratch/Fri/data")
# dataset = load_dataset("imagenet-1k")
print("2")
train_data = dataset['train']

# import random

# random_samples = random.sample(train_data, 5)

# for sample in random_samples:
#     image = sample['image']
#     label = sample['label']
#     print(image.shape)

