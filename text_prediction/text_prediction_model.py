import torch
from torch import nn
from torchmetrics.functional import accuracy, f1_score, precision, recall, perplexity, bleu_score

from model import Transformer


class GPT(nn.Module):
    """GPT model"""
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoding_layers=0, num_decoding_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        """
        :param vocab_size:
        :param d_model:
        :param nhead:
        :param num_encoding_layers:
        :param num_decoding_layers:
        :param dim_feedforward:
        :param dropout:
        """
        super(GPT, self).__init__()
        self.transformer = Transformer(d_model, nhead, num_encoding_layers, num_decoding_layers, dim_feedforward,
                                       dropout)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt, src_mask, tgt_mask)
        output = self.linear(output)
        return output

    def generate(self, src, max_len=100, max_seq_len=32):
        for _ in range(max_len):
            idx_cond = src[:, -max_seq_len:]
            output = self(idx_cond, idx_cond)  # (batch_size, seq_len, vocab_size)
            output = output[:, -1, :]  # (batch_size, vocab_size)
            probs = nn.functional.softmax(output, dim=-1)  # (batch_size, vocab_size)
            output = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            src = torch.cat([src, output], dim=1)  # (batch_size, seq_len + 1)
        return src


def calculate_loss(outputs, targets):
    B, T, C = outputs.shape
    logits = outputs.view(B * T, C)
    targets = targets.reshape(B * T)
    return torch.nn.functional.cross_entropy(logits, targets)


class TextPredictor:
    """Text predictor"""

    def __init__(self, model, tokenizer, device=None, lr=1e-4):
        """
        :param model:
        :param tokenizer:
        :param device:
        :param lr:
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.tokenizer = tokenizer
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def evaluate(self, dataloader):
        total_loss, total_metrics = 0, {}
        with torch.no_grad():
            for src in dataloader:
                if src.__class__.__name__ == 'dict':
                    src_ids = src['input_ids'].to(self.device)
                    src_mask_batch = src['attention_mask'].to(self.device)

                    tgt = src_ids[:, 1:].to(self.device)
                    src = src_ids[:, :-1].to(self.device)
                    src_mask = src_mask_batch[:, :-1].to(self.device)
                    tgt_mask = src_mask_batch[:, 1:].to(self.device)
                    # Reshape src_mask and tgt_mask
                    src_mask = src_mask.reshape(src_mask.shape[0], 1, 1, src_mask.shape[1])
                    tgt_mask = tgt_mask.reshape(tgt_mask.shape[0], 1, 1, tgt_mask.shape[1])

                    output = self.model(src, tgt, src_mask, tgt_mask)
                else:
                    tgt = src[:, 1:].to(self.device)
                    src = src[:, :-1].to(self.device)
                    output = self.model(src, tgt)

                loss = calculate_loss(output, tgt)

                # Calculate metrics
                metrics = self.metrics(output, tgt)
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v

                total_loss += loss.item()

        for k, v in total_metrics.items():
            total_metrics[k] = v / len(dataloader)

        total_loss = total_loss / len(dataloader)
        return total_loss, total_metrics

    def train(self, train_dataloader, val_dataloader, epochs=10, use_wandb=False):
        for i in range(epochs):
            total_loss, total_metrics = 0, {}
            for src in train_dataloader:
                self.optimizer.zero_grad(set_to_none=True)
                # If using huggingface tokenizer
                if src.__class__.__name__ == 'dict':
                    src_ids = src['input_ids'].to(self.device)
                    src_mask_batch = src['attention_mask'].to(self.device)

                    tgt = src_ids[:, 1:].to(self.device)
                    src = src_ids[:, :-1].to(self.device)
                    src_mask = src_mask_batch[:, :-1].to(self.device)
                    tgt_mask = src_mask_batch[:, 1:].to(self.device)
                    # Reshape src_mask and tgt_mask
                    src_mask = src_mask.reshape(src_mask.shape[0], 1, 1, src_mask.shape[1])
                    tgt_mask = tgt_mask.reshape(tgt_mask.shape[0], 1, 1, tgt_mask.shape[1])

                    output = self.model(src, tgt, src_mask, tgt_mask)
                else:
                    tgt = src[:, 1:].to(self.device)
                    src = src[:, :-1].to(self.device)
                    output = self.model(src, tgt)

                loss = calculate_loss(output, tgt)

                # Calculate metrics

                metrics = self.metrics(output, tgt)

                loss.backward()
                self.optimizer.step()

                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
                total_loss += loss.item()
            train_loss = total_loss / len(train_dataloader)

            for k, v in total_metrics.items():
                total_metrics[k] = v / len(train_dataloader)
            val_loss, val_metrics = self.evaluate(val_dataloader)
            print(f"Epoch {i + 1}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                  f"train_perplexity: {total_metrics['perplexity']:.4f} | val_perplexity: {val_metrics['perplexity']:.4f}"
                  f" | train_bleu: {total_metrics['bleu']:.4f} | val_bleu: {val_metrics['bleu']:.4f}"
                  f" | train_accuracy: {total_metrics['accuracy']:.4f} | val_accuracy: {val_metrics['accuracy']:.4f}"
                  f" | train_recall: {total_metrics['recall']:.4f} | val_recall: {val_metrics['recall']:.4f}"
                  f" | train_precision: {total_metrics['precision']:.4f} | val_precision: {val_metrics['precision']:.4f}"
                  f" | train_f1: {total_metrics['f1']:.4f} | val_f1: {val_metrics['f1']:.4f}")

            if use_wandb:
                import wandb
                wandb.log(
                    {"train_loss": train_loss, "val_loss": val_loss, "train_perplexity": total_metrics["perplexity"],
                     "val_perplexity": val_metrics["perplexity"], "train_bleu": total_metrics["bleu"],
                     "val_bleu": val_metrics["bleu"], "train_accuracy": total_metrics["accuracy"],
                     "val_accuracy": val_metrics["accuracy"], "train_recall": total_metrics["recall"],
                     "val_recall": val_metrics["recall"], "train_precision": total_metrics["precision"],
                     "val_precision": val_metrics["precision"], "train_f1": total_metrics["f1"],
                     "val_f1": val_metrics["f1"], "epoch": i + 1})

    def generate(self, context, max_len=100, max_seq_len=32):
        """Generate text given context
        :param context: str
        :param max_len: int (max length of generated text)
        :param max_seq_len: int (max length of sequence to feed into model)
        :return: str"""
        context = self.tokenizer.encode(context)
        context = torch.tensor(context).unsqueeze(0).to(self.device)
        output = self.model.generate(context, max_len, max_seq_len)
        output = output[0].cpu().numpy().tolist()
        return self.tokenizer.decode(output)

    @torch.no_grad()
    def metrics(self, output, targets):
        """Calculate metrics
        :param output: torch.Tensor
        :param targets: torch.Tensor
        :return: dict"""

        preds = torch.argmax(output, dim=-1)
        targets = targets
        target_sentence = [self.tokenizer.decode(target) for target in targets]
        pred_sentence = [self.tokenizer.decode(pred) for pred in preds]

        return {"accuracy": accuracy(preds, targets, task="multiclass", num_classes=self.tokenizer.vocab_size),
                "f1": f1_score(preds, targets, task="multiclass", num_classes=self.tokenizer.vocab_size),
                "precision": precision(preds, targets, task="multiclass", num_classes=self.tokenizer.vocab_size),
                "recall": recall(preds, targets, task="multiclass", num_classes=self.tokenizer.vocab_size),
                "perplexity": perplexity(output, targets),
                "bleu": bleu_score(pred_sentence, target_sentence)}
