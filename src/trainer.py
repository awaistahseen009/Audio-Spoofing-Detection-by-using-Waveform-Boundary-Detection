import torch
from torch.utils.data import DataLoader, random_split
from src.model import BoundaryDetectionModel
from data.audio_dataset import AudioDataset
from utils.metrics import calculate_eer
from utils.logger import setup_logger
from config.wandb_config import init_wandb, log_metrics, log_model_checkpoint, finish_wandb
import numpy as np
import os
from torch import nn

logger = setup_logger(__name__, log_type='training')

class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=1600, scale=1):
        self.warmup_steps = warmup_steps
        self.scale = scale
        super(NoamScheduler, self).__init__(optimizer)

    def get_lr(self):
        step = max(1, self._step_count)
        return [
            self.scale / (self.warmup_steps ** 0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)
            for _ in self.base_lrs
        ]

def train(model, dataloader, criterion, optimizer, scheduler, device, epoch=None):
    model.train()
    running_loss = 0.0
    all_labels, all_outputs = [], []
    for batch_idx, (audio, labels) in enumerate(dataloader, 1):
        audio = audio.to(device).squeeze(1)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(audio).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * audio.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_outputs.extend(outputs.detach().cpu().numpy())
        
        if batch_idx % 200 == 0:
            eer, _ = calculate_eer(np.array(all_labels).flatten(), np.array(all_outputs).flatten())
            batch_loss = running_loss / (batch_idx * len(audio))
            logger.info(f"Batch {batch_idx}: Running Loss = {batch_loss:.4f}, EER = {eer:.2f}%")
            
            log_metrics({
                "train/batch_loss": batch_loss,
                "train/batch_eer": eer
            })
            
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_outputs = [], []
    with torch.no_grad():
        for audio, labels in dataloader:
            audio = audio.to(device).squeeze(1)
            labels = labels.to(device)
            outputs = model(audio).squeeze(-1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * audio.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.detach().cpu().numpy())
    eer, _ = calculate_eer(np.array(all_labels).flatten(), np.array(all_outputs).flatten())
    return running_loss / len(dataloader.dataset), eer

def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    init_wandb(project_name="audio-spoofing-detection")
    
    extrinsic_dir = "new_files/Extrinsic_Partial_Fakes"
    intrinsic_dir = "new_files/Intrinsic_Partial_Fakes"
    real_dir = "audio_files/Real/training"
    label_dir = "new_files/Generated_Labels"

    audio_files = [os.path.join(extrinsic_dir, f) for f in os.listdir(extrinsic_dir) if f.endswith(".wav")] + \
                  [os.path.join(intrinsic_dir, f) for f in os.listdir(intrinsic_dir) if f.endswith(".wav")] + \
                  [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(".wav")]

    dataset = AudioDataset(audio_files, label_dir, sample_rate=16000, target_duration=6.0)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = BoundaryDetectionModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = NoamScheduler(optimizer)

    for epoch in range(10):
        logger.info(f"Epoch {epoch+1}:")
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss, eer = evaluate(model, val_loader, criterion, device)
        logger.info(f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, EER = {eer:.2f}%")

        log_metrics({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/eer": eer
        })

        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'eer': eer,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        log_model_checkpoint(checkpoint_path, epoch + 1)
    
    finish_wandb()

if __name__ == "__main__":
    main()
