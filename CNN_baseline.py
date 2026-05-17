import torch
import torch.nn as nn
import swin_transformer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=128, num_layers=1, bidirectional=True
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 6)

    def forward(self, x, lengths):
        B, T, H, W = x.shape
        x = x.view(B * T, 1, H, W)
        x = self.CNN(x)  # B*T 64
        x = x.view(B, T, x.size(-1))  # B T 64
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (ht, ct) = self.lstm(packed)
        ht = torch.cat((ht[0], ht[1]), dim=-1)
        ht = self.dropout(ht)
        outputs = self.fc(ht)
        return outputs