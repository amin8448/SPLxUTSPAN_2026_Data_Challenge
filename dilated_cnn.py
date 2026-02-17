import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold


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


class DilatedCNN(nn.Module):
    def __init__(self, n_channels=57, n_frames=240):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, dilation=4, padding=4)
        self.conv4 = nn.Conv1d(64, 32, kernel_size=3, dilation=8, padding=8)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        
        self.participant_embed = nn.Embedding(6, 16)
        
        self.fc1 = nn.Linear(32 + 16, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, pid):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = x.mean(dim=2)
        
        p = self.participant_embed(pid)
        x = torch.cat([x, p], dim=1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x).squeeze(-1)


def train_dilated_cnn(train_seq, test_seq, y, train_pid, test_pid,
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
            
            model = DilatedCNN().to(device)
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
        
        model = DilatedCNN().to(device)
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
