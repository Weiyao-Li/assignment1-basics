# LLM from Scratch

A minimal implementation of a GPT-style language model, built from the ground up in PyTorch. Covers BPE tokenization, Transformer architecture with RoPE, AdamW optimizer, and autoregressive text generation.

## Setup

```sh
pip install uv
uv run <python_file_path>
```

Or activate the virtual environment directly:

```sh
source .venv/bin/activate
```

## Usage

### Train

```sh
python3 -m cs336_basics.train \
  --train_data data/train.bin \
  --val_data data/val.bin \
  --checkpoint_dir checkpoints \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 6 \
  --num_heads 8 \
  --d_ff 1344 \
  --batch_size 32 \
  --total_steps 10000
```

### Generate

```sh
python3 -m cs336_basics.generate \
  --prompt "Once upon a time" \
  --temperature 0.8 \
  --top_k 50
```

### Tests

```sh
uv run pytest
```

## Data

```sh
mkdir -p data && cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```
