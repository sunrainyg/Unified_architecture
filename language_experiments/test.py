import torch
import argparse
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset


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
    parser.add_argument('--resume', type=bool, default=True, help='If resume from pretrained model; Set this as True when doing testing')
    
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
        model               = GPT2LMHeadModel(config)

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

