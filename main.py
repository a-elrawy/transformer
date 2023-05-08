import argparse
from data import get_shakespeare_dataloader
from text_prediction import TextPredictor, GPT
from transformers import GPT2Tokenizer


def main(args=None):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq_len', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--num_encoding_layers', type=int, default=4)
    parser.add_argument('--num_decoding_layers', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
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
    main(args)
