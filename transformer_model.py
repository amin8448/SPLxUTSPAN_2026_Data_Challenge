import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold

from preprocessing import extract_sequences, normalize_by_participant


class SequenceDataset(Dataset):
    def __init__(self, sequences, participants, targets, augment=True):
        self.sequences = sequences
        self.participants = participants
        self.targets = targets
        self.augment = augment
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = self.sequences[idx].copy()
        
        if self.augment and np.random.random() < 0.5:
            x = x + np.random.normal(0, 0.05, x.shape)
            x = x * np.random.normal(1, 0.05, (x.shape[0], 1))
        
        return (
            torch.FloatTensor(x),
            torch.LongTensor([self.participants[idx]])[0],
            torch.FloatTensor([self.targets[idx]])[0]
        )


class TransformerPredictor(nn.Module):
    def __init__(self, n_channels=57, n_frames=240, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_frames, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.participant_embed = nn.Embedding(6, 16)
        self.fc1 = nn.Linear(d_model + 16, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, pid):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.pos_enc
        x = self.transformer(x)
        x = x.mean(dim=1)
        
        p = self.participant_embed(pid)
        x = torch.cat([x, p], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x).squeeze(-1)


def train_transformer(train_seq, test_seq, y, train_pid, test_pid, 
                      n_folds=5, n_seeds=5, n_epochs=100, device='cuda'):
    oof = np.zeros(len(y))
    test_preds = np.zeros(len(test_seq))
    
    for seed in range(n_seeds):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42 + seed)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_seq)):
            torch.manual_seed(42 + seed + fold * 100)
            
            train_dataset = SequenceDataset(
                train_seq[train_idx], 
                train_pid[train_idx], 
                y[train_idx]
            )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            model = TransformerPredictor().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            
            for epoch in range(n_epochs):
                model.train()
                for bx, bp, by in train_loader:
                    bx, bp, by = bx.to(device), bp.to(device), by.to(device)
                    optimizer.zero_grad()
                    pred = model(bx, bp)
                    loss = F.mse_loss(pred, by)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                X_vl = torch.FloatTensor(train_seq[val_idx]).to(device)
                p_vl = torch.LongTensor(train_pid[val_idx]).to(device)
                oof[val_idx] += model(X_vl, p_vl).cpu().numpy()
        
        torch.manual_seed(42 + seed)
        full_dataset = SequenceDataset(train_seq, train_pid, y)
        full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
        
        model = TransformerPredictor().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        for epoch in range(n_epochs):
            model.train()
            for bx, bp, by in full_loader:
                bx, bp, by = bx.to(device), bp.to(device), by.to(device)
                optimizer.zero_grad()
                loss = F.mse_loss(model(bx, bp), by)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            X_te = torch.FloatTensor(test_seq).to(device)
            p_te = torch.LongTensor(test_pid).to(device)
            test_preds += model(X_te, p_te).cpu().numpy()
    
    return oof / n_seeds, test_preds / n_seeds
