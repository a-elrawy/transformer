import torch
from torch import nn

from model import Transformer


class GPT(nn.Module):
    """GPT model"""
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoding_layers=6, num_decoding_layers=6,
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
    def __init__(self, model, tokenizer, device=None):
        """
        :param model:
        :param tokenizer:
        :param device:
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.tokenizer = tokenizer
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def evaluate(self, dataloader):
        total_loss = 0
        with torch.no_grad():
            for src in dataloader:
                src = src[:, :-1].to(self.device)
                tgt = src[:, 1:].to(self.device)

                output = self.model(src, tgt)
                loss = calculate_loss(output, tgt)

                total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, train_dataloader, val_dataloader, epochs=10):
        for i in range(epochs):
            total_loss = 0
            for src in train_dataloader:
                self.optimizer.zero_grad(set_to_none=True)

                tgt = src[:, 1:].to(self.device)
                src = src[:, :-1].to(self.device)

                output = self.model(src, tgt)
                loss = calculate_loss(output, tgt)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_dataloader)
            val_loss = self.evaluate(val_dataloader)
            print(f"Epoch: {i+1}, Train loss: {train_loss}, Val loss: {val_loss}")

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