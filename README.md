# Text Predictor for Shakespeare's Plays

This project demonstrates how to train a text predictor using a transformer-based model on Shakespeare's plays. The goal of the project is to generate realistic text given an input context.

## Usage

To run the project, follow these steps:

1. Get dataloaders: Use the `get_shakespeare_dataloader` 
```python
from data import get_shakespeare_dataloader
train, val, test = get_shakespeare_dataloader(batch_size=16, max_seq_len=32)
```

2. Initialize model: Initialize a model with the desired hyperparameters
```python
from transformers import GPT2Tokenizer
from text_prediction import GPT
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt = GPT(vocab_size=tokenizer.vocab_size, d_model=512, nhead=8, num_encoding_layers=0, 
          num_decoding_layers=6, dim_feedforward=2048, dropout=0.1)
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