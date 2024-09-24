import os.path
import torch
from torch_geometric.data import Data
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Chgcar
import numpy as np
import json
from prepare_graph import prepare_data, calculate_weighted_properties, find_neighbors
from create_dataset import ElectronDensityPredictor, ScaledCHGCARDataset, compute_accuracy
from torch_geometric.loader import DataLoader
from pymatgen.core.sites import PeriodicSite
from prepare_graph import read_magnetic_moments
from visualization import plot_deviation_distribution, plot_chgcar


def load_model(model_name, device = 'cpu'):
    checkpoint = torch.load(model_name, map_location=device)
    model = ElectronDensityPredictor(in_channels=10, hidden_channels=64, out_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    return model

def load_standardization_parameters(filepath='standardization_parameters.json'):
    with open(filepath, 'r') as f:
        standardization_params = json.load(f)
    mean_x = torch.tensor(standardization_params['mean_x'])
    std_x = torch.tensor(standardization_params['std_x'])
    mean_y = torch.tensor([standardization_params['mean_y']])
    std_y = torch.tensor([standardization_params['std_y']])
    return mean_x, std_x, mean_y, std_y

def process_structure(structure, magnetic_moments, kpoint_radius, atom_radius, min_x, max_x, mesh, device = 'cpu'):
    sga = SpacegroupAnalyzer(structure)

    ir_kpoints_and_weights = sga.get_ir_reciprocal_mesh(mesh)
    ir_kpoints = [k[0] for k in ir_kpoints_and_weights]
    weights = [k[1] for k in ir_kpoints_and_weights]

    frac_coords = np.array(ir_kpoints)
    #grid_indices = np.floor(frac_coords * np.array(mesh)).astype(int)

    avg_properties = calculate_weighted_properties(structure, frac_coords, radius=atom_radius,
                                                   magnetic_moments=magnetic_moments)

    attributes = np.hstack([
        frac_coords,
        np.array(weights).reshape(-1, 1),
        avg_properties
    ])

    min_x_cpu = min_x.cpu().numpy()
    max_x_cpu = max_x.cpu().numpy()

    scaled_attributes = (attributes - min_x_cpu) / (max_x_cpu - min_x_cpu)

    neighbors = find_neighbors(frac_coords, radius=kpoint_radius)
    edge_index = []
    for i, kp_neighbors in enumerate(neighbors):
        for neighbor in kp_neighbors:
            edge_index.append([i, neighbor])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x = torch.tensor(scaled_attributes, dtype=torch.float).to(device)

    data = Data(x=x, edge_index=edge_index)
    return data

def unpack_kpoints(predictions, structure, mesh):

    sga = SpacegroupAnalyzer(structure)
    full_kpoints, ir_kpoint_mapping = sga.get_ir_reciprocal_mesh_map(mesh)

    full_kpoints_with_charge = np.hstack((full_kpoints, np.zeros((len(full_kpoints),2))))
    for i, ir_n in enumerate(np.unique(ir_kpoint_mapping)):
        full_kpoints_with_charge[ir_n, 3:] = predictions[i]
    for i, ir_n in enumerate(ir_kpoint_mapping):
        full_kpoints_with_charge[i, 3:] = full_kpoints_with_charge[ir_n, 3:]

    return full_kpoints_with_charge

def convert_kpoints_format(kpoints, structure, mesh):

    kpoints[:,:3] = np.floor(kpoints[:,:3] * np.array(mesh)).astype(int)
    max_indices = np.max(kpoints[:, :3], axis=0) + 1
    shape = tuple(max_indices.astype(int))

    total = np.zeros(shape)
    diff = np.zeros(shape)

    # Populate the 3D arrays
    for entry in kpoints:
        i, j, k, x, y = entry
        total[int(i), int(j), int(k)] = x
        diff[int(i), int(j), int(k)] = y

    chgcar = Chgcar(structure, {'total':total, 'diff':diff})
    return chgcar

if __name__ == "__main__":
    chgcar_file = 'samples_chgcar/Mn6V5Cr5_296_CHGCAR'

    kpoint_radius = 4
    atom_radius = 5.0

    device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = r'checkpoints/checkpoint_epoch_10.pth'
    model = load_model(model_name)

    scaling_file = "standardization_parameters.json"
    mean_x, std_x, mean_y, std_y = load_standardization_parameters(scaling_file)

    data = prepare_data(chgcar_file, kpoint_radius, atom_radius)

    data.x = (data.x - mean_x) / std_x
    data.y = (data.y - mean_y) / std_y

    with torch.no_grad():
        output = model(data)
    #output = output * (max_y - min_y) + min_y

    deviations = (output - data.y).numpy()
    print(np.mean(deviations))
    print(np.std(deviations))
    plot_deviation_distribution(np.array(deviations), 'validation')

    #chgcar = Chgcar.from_file(chgcar_file)
    #structure = chgcar.structure
    #mesh = list(chgcar.data['total'].shape)
    #full_kpoints_with_charge = unpack_kpoints(output, structure, mesh)

    #chgcar_predicted = convert_kpoints_format(full_kpoints_with_charge, structure, mesh)
    #plot_chgcar(chgcar_predicted)
    #plot_chgcar(chgcar, 'chgcar_real')
    #chgcar_predicted.write_file(f"CHGCAR")

