from data import get_shakespeare_dataloader
from text_prediction import TextPredictor, GPT
from transformers import GPT2Tokenizer


# Model Training
max_seq_len = 32
d_model = 256
nhead = 2
num_encoding_layers = 2
num_decoding_layers = 2
dim_feedforward = 64
dropout = 0.1
batch_size = 2
epochs = 10

# 1. Initialize model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt = GPT(vocab_size=tokenizer.vocab_size, d_model=d_model, nhead=nhead, num_encoding_layers=num_encoding_layers,
          num_decoding_layers=num_encoding_layers, dim_feedforward=dim_feedforward, dropout=dropout)

# 2. Get dataloaders
train_dataloader, val_dataloader, test_dataloader = get_shakespeare_dataloader(batch_size=2, max_seq_len=max_seq_len,
                                                                               tokenizer=tokenizer)


# 3. Initialize text predictor
text_predictor = TextPredictor(gpt, tokenizer)
print(sum(p.numel() for p in text_predictor.model.parameters())/1e6, 'M parameters')

# 4. Train model
text_predictor.train(train_dataloader, val_dataloader, epochs=10)

# 5. Evaluate model
test_loss = text_predictor.evaluate(test_dataloader)
print(f'Test loss: {test_loss}')

# 6. Generate text

# context = torch.zeros((1, 1), dtype=torch.long)
# x = gpt.generate(context, 100)

context = 'To be or not to be, that is the question'

print(f'Context: {context}')
print(f'Generated text: {text_predictor.generate(context, max_len=100, max_seq_len=max_seq_len)}')





