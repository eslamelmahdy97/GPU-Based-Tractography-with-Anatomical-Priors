import os
import torch

def save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path="checkpoint.pt"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"[Checkpoint] Saved at epoch {epoch+1}")

def load_checkpoint(model, optimizer, checkpoint_path="checkpoint.pt"):
    if not os.path.isfile(checkpoint_path):
        print("[Checkpoint] No checkpoint file found.")
        return None, 0  # no checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Manually move each tensor in optimizer state to GPU
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    last_epoch = checkpoint['epoch']
    train_loss = checkpoint.get('train_loss', None)
    print(f"[Checkpoint] Loaded from epoch {last_epoch+1}, train_loss={train_loss}")
    return train_loss, last_epoch