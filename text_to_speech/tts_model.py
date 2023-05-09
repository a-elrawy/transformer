import librosa
import torch
from torch import nn

from model import Transformer
from utils.audio_preprocessing import spectrogram_to_audio


class TTSModel(nn.Module):
    """TexT To Speech model based on Transformer."""

    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoding_layers=6, num_decoding_layers=6,
                 dim_feedforward=2048, dropout=0.1, input_dim=80, output_dim=80):
        """
        :param vocab_size:
        :param d_model:
        :param nhead:
        :param num_encoding_layers:
        :param num_decoding_layers:
        :param dim_feedforward:
        :param dropout:
        :param input_dim:
        :param output_dim:
        """
        super(TTSModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transformer = Transformer(d_model, nhead, num_encoding_layers, num_decoding_layers, dim_feedforward,
                                       dropout)
        self.text_emb = nn.Embedding(vocab_size, d_model)
        self.spec_emb = nn.Linear(input_dim, d_model)
        self.linear = nn.Linear(d_model, output_dim)

    def forward(self, text, spec, text_mask=None, spec_mask=None):
        """
        :param spec_mask:
        :param text_mask:
        :param text: (batch_size, seq_len)
        :param spec: (batch_size, input_dim, frames)
        :return: spec: (batch_size, output_dim, frames)
        """
        text = self.text_emb(text)  # (batch_size, seq_len, d_model)
        spec = self.spec_emb(spec)  # (batch_size, input_dim, frames, d_model)
        output = self.transformer(text, spec, text_mask, spec_mask)
        output = self.linear(output)
        return output


class TextToSpeech:
    """Text to speech model"""

    def __init__(self, model, tokenizer, device=None, lr=1e-4):
        """
        :param model:
        :param tokenizer:
        :param device:
        :param lr: learning rate
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.tokenizer = tokenizer
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def evaluate(self, dataloader):
        total_loss = 0
        with torch.no_grad():
            for text, spec in dataloader:
                text = text.to(self.device)
                spec = spec.to(self.device)
                output = self.model(text, spec)
                loss = self.mse_loss(output, spec)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, train_dataloader, val_dataloader, epochs=10, use_wandb=False):
        for i in range(epochs):
            total_loss = 0
            for text, spec in train_dataloader:
                self.optimizer.zero_grad()
                text = text.to(self.device)
                spec = spec.to(self.device)

                generated_spec = self.model(text, spec)

                # Compute the loss
                loss = self.mse_loss(generated_spec, spec)

                # Backpropagate the gradients
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                print(loss.item())
                if loss.item() < 0.2:
                    break
            train_loss = total_loss / len(train_dataloader)
            val_loss = self.evaluate(val_dataloader)
            print(f"Epoch: {i + 1}, Train loss: {train_loss}, Val loss: {val_loss}")
            if use_wandb:
                import wandb
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})

    def generate(self, text, sr=22050, max_len=1000):
        """Generate audio from text
        :param text: text to generate audio from
        :param sr: sample rate
        :param max_len: maximum length of the generated audio
        :return: audio"""
        tokens = self.tokenizer.encode(text)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)

        # Initialize audio with random noise
        spec = torch.randn(1, max_len, self.model.input_dim).to(self.device)

        with torch.no_grad():
            for i in range(max_len):
                audio_pred = self.model(tokens, spec[:, :i + 1, :])
                spec[:, i, :] = audio_pred[:, -1, :]

            # spec = self.model(tokens, spec)

        # Convert audio to waveform
        audio_pred = spec.squeeze().detach().cpu().numpy()
        audio_pred = spectrogram_to_audio(audio_pred, sr)
        return audio_pred
