import argparse
from transformers import GPT2Tokenizer


def text_prediction(args=None):
    from text_prediction import TextPredictor, GPT, get_shakespeare_dataloader
    # 1. Initialize model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt = GPT(vocab_size=tokenizer.vocab_size, d_model=args.d_model, nhead=args.nhead,
              num_encoding_layers=args.num_encoding_layers,
              num_decoding_layers=args.num_encoding_layers, dim_feedforward=args.dim_feedforward, dropout=args.dropout)

    # 2. Get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_shakespeare_dataloader(batch_size=args.batch_size,
                                                                                   max_seq_len=args.max_seq_len,
                                                                                   tokenizer=tokenizer)

    # 3. Initialize text predictor
    text_predictor = TextPredictor(gpt, tokenizer, lr=args.lr)
    print(sum(p.numel() for p in text_predictor.model.parameters()) / 1e6, 'M parameters')

    # 3.1 Initialize wandb
    if args.use_wandb:
        import wandb
        wandb.init(project='text-prediction')

    # 4. Train model
    if args.train:
        text_predictor.train(train_dataloader, val_dataloader, epochs=args.epochs, use_wandb=args.use_wandb)

    # 5. Evaluate model
    test_loss = text_predictor.evaluate(test_dataloader)
    print(f'Test loss: {test_loss}')

    # 6. Generate text
    if args.generate:
        context = args.context

        print(f'Context: {context}')
        print(f'Generated text: {text_predictor.generate(context, max_len=args.max_len, max_seq_len=args.max_seq_len)}')


def text_to_speech(args=None):
    from text_to_speech import TTSModel, TextToSpeech, get_LJ_dataloader

    # 1. Initialize model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tts_model = TTSModel(vocab_size=tokenizer.vocab_size, d_model=args.d_model, nhead=args.nhead,
                         num_encoding_layers=args.num_encoding_layers, num_decoding_layers=args.num_decoding_layers,
                         dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                         input_dim=args.input_dim, output_dim=args.output_dim)

    # 2. Get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_LJ_dataloader(batch_size=args.batch_size,
                                                                          max_seq_len=args.max_seq_len,
                                                                          tokenizer=tokenizer)

    # 3. Initialize text to speech model
    tts = TextToSpeech(tts_model, tokenizer, lr=args.lr)
    print(sum(p.numel() for p in tts.model.parameters()) / 1e6, 'M parameters')

    # 3.1 Initialize wandb
    if args.use_wandb:
        import wandb
        wandb.init(project='text-to-speech')

    # 4. Train model
    if args.train:
        tts.train(train_dataloader, val_dataloader, epochs=args.epochs, use_wandb=args.use_wandb)

    # 5. Evaluate model
    test_loss = tts.evaluate(test_dataloader)
    print(f'Test loss: {test_loss}')

    # 6. Generate audio
    if args.generate:
        context = args.context
        print(f'Context: {context}')
        print(f'Generated audio: {tts.generate(context, max_len=args.max_len)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='text_prediction')
    parser.add_argument('--max_seq_len', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--num_encoding_layers', type=int, default=4)
    parser.add_argument('--num_decoding_layers', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--input_dim', type=int, default=80)
    parser.add_argument('--output_dim', type=int, default=80)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--generate', type=bool, default=False)
    parser.add_argument('--context', type=str, default='To be or not to be')
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--use_wandb', type=bool, default=False)
    args = parser.parse_args()

    if args.task == 'text_prediction':
        text_prediction(args)
    elif args.task == 'text_to_speech':
        text_to_speech(args)
