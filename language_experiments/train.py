"""
Finetune a pretrained gpt2 model on a custom dataset.
    Original Paper and repository here: https://github.com/openai/gpt-2
    Adapted from code by nshepperd: https://github.com/nshepperd/gpt-2/blob/finetuning/train.py
"""
import os
import tqdm
import time
import argparse
import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import numpy as np

from GPT2.model import GPT2LMHeadModel
from GPT2.utils import load_weight
from GPT2.config import get_config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder
from GPT2.data import Sampler

FINETUNED_DIR = 'finetuned_models'
CHECKPOINT_DIR = os.path.join(FINETUNED_DIR, 'checkpoint')
SAMPLE_DIR = os.path.join(FINETUNED_DIR, 'samples')


def get_state_dict(model_name='117M'):
    model_path = os.path.join('pretrained_models', model_name, 'model.bin')

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None)
    else:
        print('model does not exist at: {}'.format(model_path))
        exit(0)
    return state_dict


def load_model(model, state_dict, device):
    model = load_weight(model, state_dict)
    model.to(device)
    return model


def get_latest_ckpt(ckpt_run_dir):

    if not os.path.isdir(ckpt_run_dir):
        return None

    ckpts = [ckpt for ckpt in os.listdir(ckpt_run_dir) if ckpt.endswith('.tar')]

    if len(ckpts) == 0:
        return None

    ckpts = [(ckpt, int(ckpt.split('.')[0].split('-')[1])) for ckpt in ckpts]
    ckpt, counter = max(ckpts, key=lambda tup: tup[1])
    ckpt_path = os.path.join(ckpt_run_dir, ckpt)

    return ckpt_path


def maketree(path):
    os.makedirs(path, exist_ok=True)


def load_dataset_from_huggingface(enc, limit_train, limit_val):
    dataset = load_dataset("roneneldan/TinyStories")
    
    data = dataset['train'][:limit_train]['text']
    data_val = dataset['validation'][:limit_val]['text']

    token_chunks_train, token_chunks_val = [], []

    for story in tqdm.tqdm(data):
        tokens = np.array(enc.encode(story))
        token_chunks_train.append(tokens)
    
    for story in tqdm.tqdm(data_val):
        tokens = np.array(enc.encode(story))
        token_chunks_val.append(tokens)

    return token_chunks_train, token_chunks_val

def generate_response(prompt_text, model, enc, device, max_length=250):
    # 将启动文本编码为token
    context_tokens = enc.encode(prompt_text)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用sample_sequence函数生成文本
    out = sample_sequence(
        model=model, length=max_length, context=torch.tensor(context_tokens, device=device).unsqueeze(0),
        start_token=None, batch_size=1, temperature=1.0, top_k=40, device=device
    )
    
    # 将token解码为文本
    response = enc.decode(out[0].tolist())
    return response

def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 on your custom dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', metavar='PATH', type=str, required=True,
                        help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
    parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
    parser.add_argument('--combine', metavar='CHARS', type=int, default=50000,
                        help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')

    parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.00002, help='Learning rate for Adam')
    parser.add_argument('--accumulate_gradients', metavar='N', type=int, default=1,
                        help='Accumulate gradients across N minibatches.')
    parser.add_argument('--only_train_transformer_layers', default=False, action='store_true',
                        help='Restrict training to the transformer blocks.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer. <adam|sgd>.')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Add noise to input training data to regularize against typos.')

    parser.add_argument('--top_k', type=int, default=40, help='K for top-k sampling.')
    parser.add_argument('--top_p', type=float, default=0.0, help='P for top-p sampling. Overrides top_k if set > 0.')

    parser.add_argument('--restore_from', type=str, default='latest',
                        help='Either "latest", "fresh", or a path to a checkpoint file')
    parser.add_argument('--run_name', type=str, default='run1',
                        help='Run id. Name of subdirectory in finetuned_models/')
    parser.add_argument('--sample_every', metavar='N', type=int, default=100, help='Generate samples every N steps')
    parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=1023, help='Sample this many tokens')
    parser.add_argument('--sample_num', metavar='N', type=int, default=1, help='Generate this many samples')
    parser.add_argument('--save_every', metavar='N', type=int, default=1000, help='Write a checkpoint every N steps')

    parser.add_argument('--val_dataset', metavar='PATH', type=str, default=None,
                        help='Dataset for validation loss, defaults to --dataset.')
    parser.add_argument('--val_batch_size', metavar='SIZE', type=int, default=2, help='Batch size for validation.')
    parser.add_argument('--val_batch_count', metavar='N', type=int, default=40,
                        help='Number of batches for validation.')
    parser.add_argument('--val_every', metavar='STEPS', type=int, default=0,
                        help='Calculate validation loss every STEPS steps.')

    # settings
    args = parser.parse_args()
    print(args)

    enc = get_encoder()
    config = get_config(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel(config)

    # error checking
    if args.sample_length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    if args.model_name == '345M':
        args.memory_saving_gradients = True
        if args.optimizer == 'adam':
            args.only_train_transformer_layers = True

    # select variables to update while training
    all_vars = [tensor for tensor in model.parameters()]

    # create optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(all_vars, lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(all_vars, lr=args.learning_rate)
    else:
        exit('Bad optimizer:', args.optimizer)

    # # load model
    # if args.restore_from == 'latest':
    #     ckpt_path = get_latest_ckpt(os.path.join(CHECKPOINT_DIR, args.run_name))

    #     if ckpt_path is None:
    #         state_dict = get_state_dict(args.model_name)
    #         model = load_model(model, state_dict, device)
    #         counter = 1

    #     else:
    #         ckpt = torch.load(ckpt_path)
    #         model = load_model(model, ckpt['model_state_dict'], device)
    #         optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    #         counter = ckpt['counter']

    # elif args.restore_from == 'fresh':
    #     state_dict = get_state_dict(args.model_name)
    #     model = load_model(model, state_dict, device)
    #     counter = 1

    # else:  # path to a checkpoint tar file
    #     ckpt = torch.load(args.restore_from)
    #     model = load_model(model, ckpt['model_state_dict'], device)
    #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    #     counter = ckpt['counter']
    
    ## initialize a model
    model = GPT2LMHeadModel(config).to(device)

    # load datasets
    print('load training dataset...')
    # chunks = load_dataset(enc, args.dataset, args.combine)
    chunks_train, chunks_val = load_dataset_from_huggingface(enc, limit_train=30000, limit_val=100)
    data_sampler = Sampler(chunks_train)
    print('dataset has {} tokens'.format(data_sampler.total_size))

    if args.val_every > 0:
        # Sample from validation set once with fixed seed to make
        # it deterministic during training as well as across runs.
        print('load validation dataset...')
        # val_chunks = load_dataset(enc, args.val_dataset, args.combine) if args.val_dataset else chunks
        val_data_sampler = Sampler(chunks_train, seed=1)
        val_batches = torch.tensor([[val_data_sampler.sample(256) for _ in range(args.val_batch_size)]
                                    for _ in range(args.val_batch_count)])

    def save():
        maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
        save_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'ckpt-{}.tar'.format(counter))
        torch.save({
            'counter': counter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, save_path)

    def generate_samples():
        """Generate unconditional samples."""
        print('Generating samples...')

        generated = 0
        all_text = []

        for _ in range(args.sample_num):
            out = sample_sequence(
                model=model, length=args.sample_length, context=None,
                start_token=enc.encoder['<|endoftext|>'], batch_size=1,
                temperature=1.0, top_k=args.top_k, device=device
            )

            out = out[:, :].tolist()[0]
            generated += 1
            text = enc.decode(out)
            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)
            all_text.append(text)

        maketree(os.path.join(SAMPLE_DIR, args.run_name))
        with open(os.path.join(SAMPLE_DIR, args.run_name, 'samples-{}.txt'.format(counter)), 'w') as fp:
            fp.write('\n'.join(all_text))

    def validation():
        print('Calculating validation loss...')
        losses = []
        for batch in tqdm.tqdm(val_batches):
            loss = model(batch[:, :-1].to(device), lm_labels=batch[:, 1:].to(device))
            losses.append(loss)
        v_val_loss = torch.mean(torch.tensor(losses))
        print('[{counter} | {time:2.2f}] validation loss = {loss:2.2f}'
              .format(counter=counter, time=time.time() - start_time, loss=v_val_loss))

    def sample_batch():
        return torch.tensor([data_sampler.sample(128) for _ in range(args.batch_size)])

    avg_loss = (0.0, 0.0)
    start_time = time.time()

    # training
    counter = 0
    epoches = 10
    for epoch in range(epoches):
        try:
            if counter % args.save_every == 0:
                save()
            if counter % args.sample_every == 0:
                generate_samples()
            if args.val_every > 0 and (counter % args.val_every == 0 or counter == 1):
                validation()

            if args.accumulate_gradients > 1:
                optimizer.zero_grad()

                for _ in range(args.accumulate_gradients):
                    batch = sample_batch()
                    loss = model(batch[:, :-1].to(device), lm_labels=batch[:, 1:].to(device))
                    loss.backward()
                    optimizer.step()

            else:
                optimizer.zero_grad()
                batch = sample_batch()
                loss = model(batch[:, :-1].to(device), lm_labels=batch[:, 1:].to(device))
                loss.backward()
                optimizer.step()

            avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)

            print('[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                .format(counter=counter, time=time.time() - start_time,
                        loss=loss, avg=avg_loss[0] / avg_loss[1]))

            counter += 1

        except KeyboardInterrupt:
            print('interrupt')
            save()
    
    PATH = '/om2/group/cbmm/data/gpt2_tinystories.pth'
    torch.save(model.state_dict(), PATH)
    
    model.eval()  # 将模型设置为评估模式

    # 使用函数生成响应
    prompt = "go on adventures and explore the world . one day , whiskers went for a walk and found a shiny , red automobile . he climbed inside and pretended to drive "  # 你可以更改这里的文本
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    response = generate_response(prompt, model, enc, config, device)
    print(response)
    


if __name__ == '__main__':
    main()
