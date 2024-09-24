import numpy as np
import torch
from pymatgen.io.vasp import Chgcar, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from pymatgen.core import Structure, PeriodicSite
from pymatgen.core.periodic_table import Element
import os

def find_neighbors(kpoints, radius):
    kpoints = np.array(kpoints)
    dist_matrix = np.linalg.norm(kpoints[:, np.newaxis] - kpoints, axis=2)
    neighbors = [list(np.where((dist_matrix[i] < radius) & (dist_matrix[i] > 0))[0]) for i in range(len(kpoints))]
    return neighbors

def read_magnetic_moments(outcar_file):
    outcar = Outcar(outcar_file)
    total_magnetic_moments = [site['tot'] for site in outcar.magnetization]
    return np.array(total_magnetic_moments)

def calculate_properties_with_distances(structure, kpoints, radius):
    reciprocal_lattice = structure.lattice.reciprocal_lattice
    reciprocal_lattice_matrix = reciprocal_lattice.matrix

    kpoint_coords_cartesian = reciprocal_lattice.get_cartesian_coords(kpoints)

    atomic_numbers = np.array([site.specie.Z for site in structure])
    atomic_masses = np.array([site.specie.atomic_mass for site in structure])
    electronegativities = np.array([site.specie.X for site in structure])
    atomic_radii = np.array([site.specie.atomic_radius for site in structure])

    # Calculate the number of valence electrons for s and d orbitals
    s_electrons = []
    d_electrons = []
    for site in structure:
        electronic_structure = site.specie.full_electronic_structure[-2:]

        s_electrons.append(sum(
            count for (n, orbital_type, count) in electronic_structure if orbital_type == 's'))
        d_electrons.append(sum(
            count for (n, orbital_type, count) in electronic_structure if orbital_type == 'd'))

    s_electrons = np.array(s_electrons)
    d_electrons = np.array(d_electrons)
    s_vacances = 2 - s_electrons
    d_vacances = 10 - d_electrons

    atom_coords_cartesian = structure.cart_coords

    properties_with_distances = []

    for i, kp_coords_cartesian in enumerate(kpoint_coords_cartesian):
        distances = np.linalg.norm(atom_coords_cartesian - kp_coords_cartesian, axis=1)
        mask = distances < radius

        distances = distances[mask]
        nearest_indices = np.argsort(distances)[:10]  # Take 10 nearest atoms for example

        selected_distances = distances[nearest_indices]
        selected_atomic_numbers = atomic_numbers[mask][nearest_indices]
        selected_atomic_masses = atomic_masses[mask][nearest_indices]
        selected_electronegativities = electronegativities[mask][nearest_indices]
        selected_atomic_radii = atomic_radii[mask][nearest_indices]
        selected_s_electrons = s_electrons[mask][nearest_indices]
        selected_d_electrons = d_electrons[mask][nearest_indices]
        selected_s_vacances = s_vacances[mask][nearest_indices]
        selected_d_vacances = d_vacances[mask][nearest_indices]

        # Combine distances and properties into a single array
        properties = np.concatenate((
            selected_distances,
            selected_atomic_numbers,
            selected_atomic_masses,
            selected_electronegativities,
            selected_atomic_radii,
            selected_s_electrons,
            selected_d_electrons,
            selected_s_vacances,
            selected_d_vacances
        ))

        properties_with_distances.append(properties)

    return np.array(properties_with_distances)

def prepare_data(chgcar_file, kpoint_neighbor, atom_radius):
    chgcar = Chgcar.from_file(chgcar_file)
    structure = chgcar.structure
    mesh = list(chgcar.data['total'].shape)

    sga = SpacegroupAnalyzer(structure)
    ir_kpoints = sga.get_ir_reciprocal_mesh(mesh)
    irreducible_kpoints = [k[0] for k in ir_kpoints]
    weights = [k[1] for k in ir_kpoints]

    frac_coords = np.array(irreducible_kpoints)

    # Convert fractional coordinates to grid indices
    grid_indices = np.floor(frac_coords * np.array(chgcar.data['total'].shape)).astype(int)

    total_densities = np.array([chgcar.data['total'][tuple(k)] for k in grid_indices])
    magnetic_densities = np.array([chgcar.data['diff'][tuple(k)] for k in grid_indices])

    densities = np.stack((total_densities, magnetic_densities), axis=-1)

    # Read the corresponding OUTCAR file for magnetic moments
    outcar_file = chgcar_file.replace("CHGCAR", "OUTCAR").replace("chgcar", "outcar")
    magnetic_moments = read_magnetic_moments(outcar_file)

    properties_with_distances = calculate_properties_with_distances(structure, frac_coords, radius=atom_radius)

    attributes = properties_with_distances

    edge_index = knn_graph(torch.tensor(frac_coords, dtype=torch.float), k=kpoint_neighbor)

    x = torch.tensor(attributes, dtype=torch.float)
    y = torch.tensor(densities, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def print_graph(data):
    nodes = data.x.numpy()
    print("Nodes (k-points):")
    for i, node in enumerate(nodes):
        print(f"Node {i}: {node}")

if __name__ == "__main__":
    chgcar_file = "samples_chgcar/Co8Cu4Ni4_333_CHGCAR"
    kpoint_neighbor = 2
    atom_radius = 5.00
    data = prepare_data(chgcar_file, kpoint_neighbor, atom_radius)
    print_graph(data)
    #print(data.y)
