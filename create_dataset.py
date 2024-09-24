import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, GATConv
import multiprocessing
from torch.cuda.amp import GradScaler, autocast
from prepare_graph import prepare_data
from visualization import plot_deviation_distribution
import json
import torch.onnx

num_workers = multiprocessing.cpu_count() // 2


class CHGCARDataset(Dataset):
    def __init__(self, root_dir, kpoint_neighbor=2, atom_radius=3.0, transform=None, pre_transform=None, device='cpu'):
        super(CHGCARDataset, self).__init__(root_dir, transform, pre_transform)
        self.kpoint_neighbor = kpoint_neighbor
        self.atom_radius = atom_radius
        self.root_dir = root_dir
        self.device = device

    @property
    def raw_file_names(self):
        return os.listdir(self.root_dir)

    @property
    def processed_file_names(self):
        return []  # Not used in this example

    def len(self):
        return len(self.raw_file_names)

    def get(self, idx):
        chgcar_file = os.path.join(self.root_dir, self.raw_file_names[idx])
        data = prepare_data(chgcar_file, self.kpoint_neighbor, self.atom_radius)
        return data


class ScaledCHGCARDataset(CHGCARDataset):
    def __init__(self, root_dir, kpoint_neighbor, atom_radius, min_x, max_x, min_y, max_y, transform=None,
                 pre_transform=None, device='cpu'):
        super(ScaledCHGCARDataset, self).__init__(root_dir, kpoint_neighbor, atom_radius, transform, pre_transform,
                                                  device)
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def get(self, idx):
        data = super(ScaledCHGCARDataset, self).get(idx)
        # Data should be on the CPU initially for pin_memory
        data.x = (data.x - self.min_x) / (self.max_x - self.min_x)
        data.y = (data.y - self.min_y) / (self.max_y - self.min_y)
        return data


class StandardizedCHGCARDataset(CHGCARDataset):
    def __init__(self, root_dir, kpoint_neighbor, atom_radius, mean_x, std_x, mean_y, std_y, transform=None,
                 pre_transform=None, device='cpu'):
        super(StandardizedCHGCARDataset, self).__init__(root_dir, kpoint_neighbor, atom_radius, transform, pre_transform,
                                                        device)
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y

    def get(self, idx):
        data = super(StandardizedCHGCARDataset, self).get(idx)
        data.x = (data.x - self.mean_x) / self.std_x
        data.y = (data.y - self.mean_y) / self.std_y
        return data

class ElectronDensityPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, heads=8, dropout=0.3):
        super(ElectronDensityPredictor, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.lin1 = torch.nn.Linear(hidden_channels * heads, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.dropout(x)
        x = F.elu(self.lin1(x))
        x = self.lin2(x)
        return x #F.relu(x)


class ExponentialDistanceLoss(torch.nn.Module):
    def __init__(self, importance_weights):
        super(ExponentialDistanceLoss, self).__init__()
        self.importance_weights = importance_weights
        self.mse_loss = torch.nn.MSELoss(reduction='none')  # No reduction to get the individual losses

    def forward(self, predictions, targets, distances):
        distances = distances / max(distances)
        weights = torch.exp(-3*distances)  # Exponential decline as a function of distance
        weights = weights.unsqueeze(1)  # Ensure weights has the same shape as predictions and targets
        targets = targets.unsqueeze(1)

        mse_loss = self.mse_loss(predictions, targets)
        weighted_mse_loss = weights * mse_loss

        # Apply different importance weights to each column
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            weighted_mse_loss[:, 0] *= self.importance_weights[0]
            weighted_mse_loss[:, 1] *= self.importance_weights[1]
        else:
            weighted_mse_loss *= self.importance_weights[0]

        return weighted_mse_loss.mean()

class InverseDistanceWeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super(InverseDistanceWeightedMSELoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none')

    def forward(self, predictions, targets, distances):
        weights = 1 / (distances + 1e-6)  # Avoid division by zero
        weights = weights / weights.max()  # Normalize weights to range [0, 1]
        weights = weights.unsqueeze(1)
        mse_loss = self.mse_loss(predictions, targets)
        weighted_mse_loss = weights * mse_loss
        return weighted_mse_loss.mean()

def compute_min_max(dataset, batch_size=32, device='cuda'):
    min_x = torch.full((dataset[0].x.shape[1],), float('inf'))
    max_x = torch.full((dataset[0].x.shape[1],), float('-inf'))
    min_y = torch.full((1,), float('inf'))
    max_y = torch.full((1,), float('-inf'))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for data in dataloader:
        data = data.to(device)
        batch_min_x, _ = torch.min(data.x, dim=0)
        batch_max_x, _ = torch.max(data.x, dim=0)
        min_x = torch.min(min_x, batch_min_x.cpu())
        max_x = torch.max(max_x, batch_max_x.cpu())
        batch_min_y, _ = torch.min(data.y, dim=0)
        batch_max_y, _ = torch.max(data.y, dim=0)
        min_y = torch.min(min_y, batch_min_y.cpu())
        max_y = torch.max(max_y, batch_max_y.cpu())

    return min_x, max_x, min_y, max_y


def calculate_mean_std(dataset, batch_size=32, device='cpu'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mean_x = 0.0
    std_x = 0.0
    mean_y = 0.0
    std_y = 0.0
    n_samples = 0

    for data in loader:
        data = data.to(device)
        n_samples += data.x.size(0)
        mean_x += data.x.mean(dim=0)
        std_x += data.x.std(dim=0)
        mean_y += data.y.mean()
        std_y += data.y.std()

    mean_x /= len(loader)
    std_x /= len(loader)
    mean_y /= len(loader)
    std_y /= len(loader)

    return mean_x, std_x, mean_y, std_y

def train_model(model, train_loader, optimizer, criterion, scaler, epochs, device, save_every=10, save_dir='checkpoints', early_stopping_patience=20):
    model.train()
    patience_counter = 0
    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        deviations = []
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(data)
                loss = criterion(output, data.y, data.x[:,-1])
                accuracy, deviation = compute_accuracy(output, data.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            total_accuracy += accuracy
            deviations.extend(deviation.flatten())  # Flatten deviations before adding to list
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy*100:.2f}')
        #plot_deviation_distribution(np.array(deviations), epoch + 1)
        np.savetxt(f'testing/{epoch + 1}.dat', deviations)

        # Save the model every `save_every` epochs
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, directory=save_dir)

        if avg_loss < 0.0001:
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    # Save the final model at the end of training
    save_checkpoint(model, optimizer, epochs, directory=save_dir, filename_prefix='final_model')


def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    deviations = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y, data.x[:,-1])
            accuracy, deviation = compute_accuracy(output, data.y)
            total_loss += loss.item()
            total_accuracy += accuracy
            deviations.extend(deviation.flatten())  # Flatten deviations before adding to list
    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)
    #plot_deviation_distribution(np.array(deviations), 'Test')
    np.savetxt(f'testing/Test.dat', deviations)
    return avg_loss, avg_accuracy

def compute_accuracy(pred_y, y):
    pred_y = pred_y.detach().cpu().squeeze()
    y = y.detach().cpu()
    deviation = np.abs((y - pred_y).numpy())
    accuracy = np.count_nonzero(deviation < 0.05) / (2 * y.size(0))
    return accuracy, deviation


def save_checkpoint(model, optimizer, epoch, directory='checkpoints', filename_prefix='checkpoint'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    checkpoint_path = os.path.join(directory, f'{filename_prefix}_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)


def save_scaling_parameters(min_x, max_x, min_y, max_y, filepath='scaling_parameters.json'):
    scaling_params = {
        'min_x': min_x.tolist(),
        'max_x': max_x.tolist(),
        'min_y': min_y.tolist(),
        'max_y': max_y.tolist()
    }
    with open(filepath, 'w') as f:
        json.dump(scaling_params, f)
    print(f'Scaling parameters saved to {filepath}')


def load_scaling_parameters(filepath='scaling_parameters.json'):
    with open(filepath, 'r') as f:
        scaling_params = json.load(f)
    min_x = torch.tensor(scaling_params['min_x'])
    max_x = torch.tensor(scaling_params['max_x'])
    min_y = torch.tensor([scaling_params['min_y']])
    max_y = torch.tensor([scaling_params['max_y']])
    return min_x, max_x, min_y, max_y

def save_standardization_parameters(mean_x, std_x, mean_y, std_y, filepath='standardization_parameters.json'):
    standardization_params = {
        'mean_x': mean_x.tolist(),
        'std_x': std_x.tolist(),
        'mean_y': mean_y.tolist(),
        'std_y': std_y.tolist()
    }
    with open(filepath, 'w') as f:
        json.dump(standardization_params, f)
    print(f'Standardization parameters saved to {filepath}')

def load_standardization_parameters(filepath='standardization_parameters.json'):
    with open(filepath, 'r') as f:
        standardization_params = json.load(f)
    mean_x = torch.tensor(standardization_params['mean_x'])
    std_x = torch.tensor(standardization_params['std_x'])
    mean_y = torch.tensor([standardization_params['mean_y']])
    std_y = torch.tensor([standardization_params['std_y']])
    return mean_x, std_x, mean_y, std_y



if __name__ == "__main__":
    root_dir = "samples_subset_chgcar"
    batch_size = 4
    epochs = 100
    learning_rate = 1e-4
    kpoint_neighbor = 3
    atom_radius = 5.0
    hidden_channels = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = CHGCARDataset(root_dir, kpoint_neighbor, atom_radius, device=device)
    dataset_len = len(dataset)

    scaling_file = "standardization_parameters.json"
    if os.path.exists(scaling_file):
        print("Loading standardization parameters from file.")
        mean_x, std_x, mean_y, std_y = load_standardization_parameters(scaling_file)
    else:
        print("Calculating standardization parameters.")
        mean_x, std_x, mean_y, std_y = calculate_mean_std(dataset, batch_size=32, device=device)
        save_standardization_parameters(mean_x, std_x, mean_y, std_y, scaling_file)


    # Create scaled dataset
    scaled_dataset = StandardizedCHGCARDataset(root_dir, kpoint_neighbor, atom_radius, mean_x, std_x, mean_y, std_y,
                                         device=device)

    test_data = scaled_dataset[0].x

    train_size = int(0.8 * len(scaled_dataset))
    test_size = len(scaled_dataset) - train_size
    print(f'Train size: {train_size}')
    print(f'Test size: {test_size}')

    train_dataset, test_dataset = torch.utils.data.random_split(scaled_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=True)

    model = ElectronDensityPredictor(in_channels=scaled_dataset[0].num_features, hidden_channels=hidden_channels,
                                     out_channels=1).to(device)

    criterion = ExponentialDistanceLoss(importance_weights = [1, 0.3])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()


    train_model(model, train_loader, optimizer, criterion, scaler, epochs, device, save_every=5, save_dir='checkpoints')

    test_loss, test_accuracy = test_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}')
