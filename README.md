# Text Predictor for Shakespeare's Plays

This project demonstrates how to train a text predictor using a transformer-based model on Shakespeare's plays. The goal of the project is to generate realistic text given an input context.

## Requirements
- Python 3.8
- [PyTorch](https://pytorch.org/) 2.0.0
- [Transformers](https://huggingface.co/transformers/) 4.28.1 
- [WandB](https://wandb.ai/) 0.15.2

Install the required packages using the following command:
```bash
pip install -r requirements.txt
```


## Usage

To run the project, follow these steps:

1. Initialize model: Initialize a model with the desired hyperparameters
```python
from transformers import GPT2Tokenizer
from text_prediction import GPT
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt = GPT(vocab_size=tokenizer.vocab_size, d_model=512, nhead=8, num_encoding_layers=0, 
          num_decoding_layers=6, dim_feedforward=2048, dropout=0.1)
```

2. Get dataloaders: Use the `get_shakespeare_dataloader` 
```python
from data import get_shakespeare_dataloader
train, val, test = get_shakespeare_dataloader(batch_size=16, max_seq_len=32,
                                              tokenizer=tokenizer)
```



3. Initialize text predictor: Initialize a `TextPredictor` object with the model and a tokenizer.
```python
from text_prediction import TextPredictor
text_predictor = TextPredictor(gpt, tokenizer)
```

5. Train model.
```python
text_predictor.train(train_dataloader, val_dataloader, epochs=epochs)
```

7. Evaluate model.
```python
test_loss = text_predictor.evaluate(test_dataloader)
print(f'Test loss: {test_loss}')
```

9. Generate text: Generate text using the `generate` method of the `TextPredictor`.
```python
context = 'ROMEO:'
generated_text = text_predictor.generate(context, max_length=100)
print(generated_text)
```

## CLI

The project can also be run from the command line. To train a model, run the following command:
```bash
python main.py --batch_size 16 --max_seq_len 32 --epochs 10 --lr 0.0001 --num_encoding_layers 0 --num_decoding_layers 6 --dim_feedforward 2048 --dropout 0.1
```

To generate text, run the following command:
```bash
python main.py --generate --context 'ROMEO:' --max_length 100
```

### Arguments
- `--nhead`: Number of attention heads in the model.
- `--d_model`: Dimension of the embedding vector in the model.
- `--num_encoding_layers`: Number of encoding layers in the model.
- `--num_decoding_layers`: Number of decoding layers in the model.
- `--dim_feedforward`: Dimension of the feedforward network in the model.

- `--max_seq_len`: Maximum sequence length for training and evaluation.
- `--train`: Flag to train model.
- `--batch_size`: Batch size for training and evaluation.
- `--lr`: Learning rate for training.
- `--epochs`: Number of epochs to train for.
- `--dropout`: Dropout probability for the model.
- `--generate`: Flag to generate text.
- `--context`: Context for text generation.
- `--max_length`: Maximum length of generated text.
- `--use_wandb`: Flag to use wandb for logging.