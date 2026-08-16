import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from data import get_datasets
from model import SudokuMNISTModel

def train_epoch(model, loader, optimizer, criterions, device):
    model.train()
    crit_cls, crit_reg = criterions

    total_loss_accum, missing_loss_accum, sort_loss_accum, sum_loss_accum = 0, 0, 0, 0
    missing_correct, sort_correct, sum_mae = 0, 0, 0
    num_samples = 0

    for batch in loader:
        images = batch['image'].to(device)
        missing_labels = batch['missing_digit'].to(device)
        sort_labels = batch['sorted_labels'].to(device)
        sum_labels = batch['sum_labels'].to(device)

        optimizer.zero_grad()
        out_missing, out_sort, out_sum = model(images)

        # Calculate losses
        loss_missing = crit_cls(out_missing, missing_labels)
        loss_sort = crit_cls(out_sort.reshape(-1, 3), sort_labels.reshape(-1))
        loss_sum = crit_reg(out_sum, sum_labels)

        loss = loss_missing + 2*loss_sort + loss_sum
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        batch_size = images.size(0)
        num_samples += batch_size

        total_loss_accum += loss.item() * batch_size
        missing_loss_accum += loss_missing.item() * batch_size
        sort_loss_accum += loss_sort.item() * batch_size
        sum_loss_accum += loss_sum.item() * batch_size

        missing_correct += (out_missing.argmax(1) == missing_labels).sum().item()
        sort_correct += (out_sort.argmax(2) == sort_labels).sum().item() / 6.0
        sum_mae += torch.abs(out_sum - sum_labels).sum().item() / 6.0

    return {
        "loss": total_loss_accum / num_samples,
        "miss_loss": missing_loss_accum / num_samples, "miss_acc": missing_correct / num_samples,
        "sort_loss": sort_loss_accum / num_samples, "sort_acc": sort_correct / num_samples,
        "sum_loss": sum_loss_accum / num_samples, "sum_mae": sum_mae / num_samples
    }

@torch.no_grad()
def eval_epoch(model, loader, criterions, device):
    model.eval()
    crit_cls, crit_reg = criterions

    total_loss_accum, missing_loss_accum, sort_loss_accum, sum_loss_accum = 0, 0, 0, 0
    missing_correct, sort_correct, sum_mae = 0, 0, 0
    num_samples = 0

    for batch in loader:
        images = batch['image'].to(device)
        missing_labels = batch['missing_digit'].to(device)
        sort_labels = batch['sorted_labels'].to(device)
        sum_labels = batch['sum_labels'].to(device)

        out_missing, out_sort, out_sum = model(images)

        loss_missing = crit_cls(out_missing, missing_labels)
        loss_sort = crit_cls(out_sort.reshape(-1, 3), sort_labels.reshape(-1))
        loss_sum = crit_reg(out_sum, sum_labels)

        loss = loss_missing + loss_sort + loss_sum

        batch_size = images.size(0)
        num_samples += batch_size

        total_loss_accum += loss.item() * batch_size
        missing_loss_accum += loss_missing.item() * batch_size
        sort_loss_accum += loss_sort.item() * batch_size
        sum_loss_accum += loss_sum.item() * batch_size

        missing_correct += (out_missing.argmax(1) == missing_labels).sum().item()
        sort_correct += (out_sort.argmax(2) == sort_labels).sum().item() / 6.0
        sum_mae += torch.abs(out_sum - sum_labels).sum().item() / 6.0

    return {
        "loss": total_loss_accum / num_samples,
        "miss_loss": missing_loss_accum / num_samples, "miss_acc": missing_correct / num_samples,
        "sort_loss": sort_loss_accum / num_samples, "sort_acc": sort_correct / num_samples,
        "sum_loss": sum_loss_accum / num_samples, "sum_mae": sum_mae / num_samples
    }

def plot_training_history(history, test_res):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.tight_layout(pad=5.0)

    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-o', label='Train Total Loss')
    ax.plot(epochs, history['val_loss'], 'r-s', label='Val Total Loss')
    ax.axhline(test_res['loss'], color='purple', linestyle='--', linewidth=2, label=f"Test Loss: {test_res['loss']:.4f}")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    ax.legend()

    # 2. Missing Digit Task
    ax = axes[0, 1]
    ax.plot(epochs, history['train_miss_loss'], 'b-o', label='Train Loss')
    ax.plot(epochs, history['train_miss_acc'], 'g-^', label='Train Acc')
    ax.plot(epochs, history['val_miss_loss'], 'r-s', label='Val Loss')
    ax.plot(epochs, history['val_miss_acc'], 'orange', marker='D', label='Val Acc')
    ax.axhline(test_res['miss_loss'], color='darkblue', linestyle='--', alpha=0.6, label=f"Test Loss: {test_res['miss_loss']:.4f}")
    ax.axhline(test_res['miss_acc'], color='green', linestyle='--', alpha=0.6, label=f"Test Acc: {test_res['miss_acc']:.4f}")
    ax.set_title("Missing Digit Task")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Accuracy")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    ax.legend(loc='lower center')

    # 3. Sort Order Task
    ax = axes[1, 0]
    ax.plot(epochs, history['train_sort_loss'], 'b-o', label='Train Loss')
    ax.plot(epochs, history['train_sort_acc'], 'g-^', label='Train Acc')
    ax.plot(epochs, history['val_sort_loss'], 'r-s', label='Val Loss')
    ax.plot(epochs, history['val_sort_acc'], 'orange', marker='D', label='Val Acc')
    ax.axhline(test_res['sort_loss'], color='darkblue', linestyle='--', alpha=0.6, label=f"Test Loss: {test_res['sort_loss']:.4f}")
    ax.axhline(test_res['sort_acc'], color='green', linestyle='--', alpha=0.6, label=f"Test Acc: {test_res['sort_acc']:.4f}")
    ax.set_title("Sort Order Task")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Accuracy")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    ax.legend(loc='lower right')

    # 4. Sum Prediction Task
    ax = axes[1, 1]
    ax.plot(epochs, history['train_sum_loss'], 'b-o', label='Train Loss')
    ax.plot(epochs, history['train_sum_mae'], 'g-^', label='Train MAE')
    ax.plot(epochs, history['val_sum_loss'], 'r-s', label='Val Loss')
    ax.plot(epochs, history['val_sum_mae'], 'orange', marker='D', label='Val MAE')
    ax.axhline(test_res['sum_loss'], color='darkblue', linestyle='--', alpha=0.6, label=f"Test Loss: {test_res['sum_loss']:.4f}")
    ax.axhline(test_res['sum_mae'], color='green', linestyle='--', alpha=0.6, label=f"Test MAE: {test_res['sum_mae']:.4f}")
    ax.set_title("Sum Prediction Task")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / MAE")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    ax.legend(loc='upper right')

    plt.savefig("training_history.png", dpi=300, bbox_inches='tight')
    print("\nTraining plots saved successfully as 'training_history.png'")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    train_ds, val_ds, test_ds = get_datasets("./data")
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = SudokuMNISTModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.004)

    # Loss functions definition
    criterions = (nn.CrossEntropyLoss(), nn.MSELoss())

    epochs = 8
    history = {k: [] for k in ['train_loss', 'val_loss', 'train_miss_loss', 'val_miss_loss',
                               'train_miss_acc', 'val_miss_acc', 'train_sort_loss', 'val_sort_loss',
                               'train_sort_acc', 'val_sort_acc', 'train_sum_loss', 'val_sum_loss',
                               'train_sum_mae', 'val_sum_mae']}

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterions, device)
        val_metrics = eval_epoch(model, val_loader, criterions, device)

        print(f"Epoch {epoch}/{epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f} | MissAcc: {train_metrics['miss_acc']:.4f} | SortAcc: {train_metrics['sort_acc']:.4f} | SumMAE: {train_metrics['sum_mae']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | MissAcc: {val_metrics['miss_acc']:.4f} | SortAcc: {val_metrics['sort_acc']:.4f} | SumMAE: {val_metrics['sum_mae']:.4f}")

        # Append histories
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])

        history['train_miss_loss'].append(train_metrics['miss_loss'])
        history['train_miss_acc'].append(train_metrics['miss_acc'])
        history['val_miss_loss'].append(val_metrics['miss_loss'])
        history['val_miss_acc'].append(val_metrics['miss_acc'])

        history['train_sort_loss'].append(train_metrics['sort_loss'])
        history['train_sort_acc'].append(train_metrics['sort_acc'])
        history['val_sort_loss'].append(val_metrics['sort_loss'])
        history['val_sort_acc'].append(val_metrics['sort_acc'])

        history['train_sum_loss'].append(train_metrics['sum_loss'])
        history['train_sum_mae'].append(train_metrics['sum_mae'])
        history['val_sum_loss'].append(val_metrics['sum_loss'])
        history['val_sum_mae'].append(val_metrics['sum_mae'])

    # Test phase and plotting [cite: 21, 24]
    print("\nEvaluating on Test Set...")
    test_metrics = eval_epoch(model, test_loader, criterions, device)
    print(f"Test - Loss: {test_metrics['loss']:.4f} | MissAcc: {test_metrics['miss_acc']:.4f} | SortAcc: {test_metrics['sort_acc']:.4f} | SumMAE: {test_metrics['sum_mae']:.4f}")

    plot_training_history(history, test_metrics)
    torch.save(model.state_dict(), 'sudokunet_weights.pth')