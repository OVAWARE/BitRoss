import torch
import torch.nn as nn
from transformers import BertModel

LATENT_DIM = 128
HIDDEN_DIM = 256
IMAGE_SIZE = 16


class TextEncoder(nn.Module):
    """BERT CLS projection. Freeze the backbone by default — fine-tuning BERT
    dominates wall time and often hurts small pixel-art datasets."""

    def __init__(self, hidden_size, output_size, freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, output_size)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Frozen BERT can run without building a full autograd graph
        if not any(p.requires_grad for p in self.bert.parameters()):
            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]
        return self.fc(pooled)


class CVAE(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, HIDDEN_DIM),
        )

        self.fc_mu = nn.Linear(HIDDEN_DIM + HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM + HIDDEN_DIM, LATENT_DIM)

        self.decoder_input = nn.Linear(LATENT_DIM + HIDDEN_DIM, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def encode(self, x, c):
        x = self.encoder(x)
        x = torch.cat([x, c], dim=1)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z, c):
        z = torch.cat([z, c], dim=1)
        x = self.decoder_input(z)
        x = x.view(-1, 128, 4, 4)
        return self.decoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
