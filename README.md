# Text Prediction and Text-to-Speech using Transformer

This project includes two modules for text prediction and text-to-speech using the transformer-based model. The text_prediction module generates text predictions based on a given input context, while the text_to_speech module generates human-like speech from text input.
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
### Text Prediction
```python
from text_prediction import TextPredictor, GPT, get_shakespeare_dataloader
from transformers import GPT2Tokenizer

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT(vocab_size=tokenizer.vocab_size)

# Initialize text predictor
text_predictor = TextPredictor(model, tokenizer)

# Get dataloader
train_loader, val_loader, _ = get_shakespeare_dataloader(batch_size=16, max_seq_len=32, tokenizer=tokenizer)

# Train model
text_predictor.train(train_loader, val_loader, epochs=10)

# Generate text
text_predictor.generate('ROMEO:', max_len=100, max_seq_len=32)
```

### Text-to-Speech
```python
from text_to_speech import TextToSpeech, TTSModel, get_LJ_dataloader
from transformers import GPT2Tokenizer

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TTSModel(vocab_size=tokenizer.vocab_size)

# Initialize text to speech
text_to_speech = TextToSpeech(model, tokenizer)

# Get dataloader
train_loader, val_loader, _ = get_LJ_dataloader(batch_size=16, max_seq_len=32, tokenizer=tokenizer)

# Train model
text_to_speech.train(train_loader, val_loader, epochs=10)

# Generate speech
text_to_speech.generate('input text', max_len=100)

```

## CLI

The project can also be run from the command line. To train a model, run the following command:
```bash
python main.py --task "text_prediction" --batch_size 16 --max_seq_len 32 --epochs 10 --lr 0.0001 --num_encoding_layers 0 --num_decoding_layers 6 --dim_feedforward 2048 --dropout 0.1
```

To generate text, run the following command:
```bash
python main.py --task "text_prediction" --generate --context 'ROMEO:' --max_length 100
```

### Arguments
- `--task`: Task to perform. Can be `text_prediction` or `text_to_speech`.
- `--nhead`: Number of attention heads in the model.
- `--d_model`: Dimension of the embedding vector in the model.
- `--num_encoding_layers`: Number of encoding layers in the model.
- `--num_decoding_layers`: Number of decoding layers in the model.
- `--dim_feedforward`: Dimension of the feedforward network in the model.
- `--input_dim`: Dimension of the input to the audio model.
- `--output_dim`: Dimension of the output of the audio model.
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