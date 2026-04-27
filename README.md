# Bigram Language Model

A minimal character-level language model built with PyTorch, based on Andrej Karpathy's ["Building a GPT from Scratch"](https://www.youtube.com/watch?v=kCc8FmE1O4w) video series.

## Overview

This project implements a **Bigram Language Model** — the simplest form of a neural language model that predicts the next character based only on the previous character. It demonstrates the foundational concepts of:

- Character-level tokenization
- Embedding tables for vocabulary lookup
- Cross-entropy loss optimization
- Text generation via sampling

## Project Structure

```
ng-video-lecture/
├── bigram.py      # Main training script
├── gpt.py         # (Not covered in this module)
├── input.txt      # Training data (Shakespeare text)
├── more.txt       # Additional training data
└── README.md      # This file
```

## Requirements

- Python 3.8+
- PyTorch

Install dependencies:

```bash
pip install torch
```

## Usage

Run the training script:

```bash
python bigram.py
```

The model will:
1. Load and tokenize the text from `input.txt`
2. Train for 3000 iterations
3. Print training/validation loss every 300 steps
4. Generate and print sample text at the end

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Number of parallel sequences processed |
| `block_size` | 8 | Maximum context length for predictions |
| `max_iters` | 3000 | Total training iterations |
| `learning_rate` | 1e-2 | Optimizer learning rate |
| `eval_interval` | 300 | Loss evaluation frequency |

## How It Works

### 1. Tokenization
The text is converted from characters to integers using a simple character-level vocabulary:

```python
chars = sorted(list(set(text)))  # All unique characters
stoi = { ch:i for i,ch in enumerate(chars) }  # char → int
itos = { i:ch for i,ch in enumerate(chars) }  # int → char
```

### 2. Model Architecture
The `BigramLanguageModel` uses an embedding table that directly maps each character to logits for predicting the next character:

```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

### 3. Training
The model is trained using cross-entropy loss between predicted and actual next characters:

```python
loss = F.cross_entropy(logits, targets)
```

### 4. Generation
During inference, the model samples from the probability distribution:

```python
idx_next = torch.multinomial(probs, num_samples=1)
```

## Sample Output

After training, the model generates text like:

```
od nos CAy go ghanoray t, co haringoudrou clethe k,LARof fr werar,
Is fa!


Thilemel cia h hmboomyorarifrcitheviPO, tle dst f qur'dig t cof boddo y t o ar p
```

> **Note:** The output appears garbled because a bigram model only looks at the previous single character. This is intentional — it's a baseline model. See `gpt.py` for a more advanced implementation.

## Learning Resources

- [Andrej Karpathy's YouTube Series](https://www.youtube.com/watch?v=kCc8FmE1O4w)
- [GPT Repository](https://github.com/karpathy/ng-video-lecture)
- [PyTorch Documentation](https://pytorch.org/docs/)

## License

MIT License — feel free to use this for learning and experimentation.