import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from dataset import fODFTOMDataset
from model import SimpleMLP
from loss import loss_fn
from utils import save_checkpoint, load_checkpoint


def train_one_epoch(model, dataloader, loss_fn, optimizer, nu, device):
    model.train()
    running_loss = 0.0
    count = 0

    for batch_cpu in dataloader:
        batch_gpu = batch_cpu.to(device)

        outputs = model(batch_gpu)
        loss = loss_fn(batch_gpu, outputs, nu)
        # Reset the gradients of all parameters in the optimizer to zero
        optimizer.zero_grad()
        # Compute the gradients of the loss function
        loss.backward()
        # Update the model parameters
        optimizer.step()

        running_loss += loss.item()
        count += 1

    epoch_loss = running_loss / count if count > 0 else 0.0
    return epoch_loss

def train_model(model, train_loader, loss_fn, optimizer, nu, num_epochs, device,
                checkpoint_path="checkpoint.pt"):

    # Try to load from checkpoint if it exists
    old_train_loss,  start_epoch = load_checkpoint(model, optimizer,checkpoint_path)
    best_train_loss = float('inf') if old_train_loss is None else old_train_loss
    
    # Move model to correct device
    model.to(device)
    
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, nu, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.6f}")

        # Log to Weights & Biases
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
        })

        save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
        
if __name__ == "__main__":
    wandb.init(project="GPU prior knowledge low-rank approx",
               entity="eslamelmahdy97-university-of-bonn")

    num_epochs = 50
    root_dir = "/home/s65eelma/HCP_TractSeg"

    
    full_dataset = fODFTOMDataset(root_dir)
    device = 'cuda'
    

    train_loader = DataLoader(
        full_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2, # speed up the data loading process
        pin_memory=True # Optimizes data transfer to the GPU
    )

    
    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    nu = 0.01
    
    # Train model with checkpointing
    train_model(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        nu=nu,
        num_epochs=num_epochs,
        device=device,
        checkpoint_path="checkpoint2.pt"
    )
