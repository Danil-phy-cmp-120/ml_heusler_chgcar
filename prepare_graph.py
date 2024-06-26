import numpy as np
import torch
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def find_neighbors(kpoints, radius):
    """
    Function to find neighbors within a given radius using a vectorized approach.
    """
    kpoints = np.array(kpoints)
    dist_matrix = np.linalg.norm(kpoints[:, np.newaxis] - kpoints, axis=2)
    neighbors = [list(np.where((dist_matrix[i] < radius) & (dist_matrix[i] > 0))[0]) for i in range(len(kpoints))]
    return neighbors


def calculate_weighted_properties(structure, kpoints, radius, epsilon=1e-10):
    """
    Function to calculate weighted properties (atomic number, mass, valence, electronegativity, atomic radius)
    based on the distance to atoms in reciprocal space.
    """
    reciprocal_lattice = structure.lattice.reciprocal_lattice
    reciprocal_lattice_matrix = reciprocal_lattice.matrix

    kpoint_coords_cartesian = reciprocal_lattice.get_cartesian_coords(kpoints)

    atomic_numbers = np.array([site.specie.Z for site in structure])
    atomic_masses = np.array([site.specie.atomic_mass for site in structure])
    valence_electrons = np.array([site.specie.valence[1] for site in structure])
    electronegativities = np.array([site.specie.X for site in structure])
    atomic_radii = np.array([site.specie.atomic_radius for site in structure])

    atom_coords_cartesian = structure.cart_coords

    avg_properties = np.zeros((len(kpoints), 5))

    for i, kp_coords_cartesian in enumerate(kpoint_coords_cartesian):
        kp_coords_reciprocal = np.dot(kp_coords_cartesian, np.linalg.inv(reciprocal_lattice_matrix))
        atom_coords_reciprocal = np.dot(atom_coords_cartesian, np.linalg.inv(reciprocal_lattice_matrix))
        distances = np.linalg.norm(atom_coords_reciprocal - kp_coords_reciprocal, axis=1)
        mask = distances < radius
        weights = np.zeros_like(distances)
        weights[mask] = 1 / (distances[mask] + epsilon)

        if np.any(mask):
            avg_properties[i, 0] = np.average(atomic_numbers[mask], weights=weights[mask])
            avg_properties[i, 1] = np.average(atomic_masses[mask], weights=weights[mask])
            avg_properties[i, 2] = np.average(valence_electrons[mask], weights=weights[mask])
            avg_properties[i, 3] = np.average(electronegativities[mask], weights=weights[mask])
            avg_properties[i, 4] = np.average(atomic_radii[mask], weights=weights[mask])

    return avg_properties


def prepare_data(chgcar_file, kpoint_radius, atom_radius):
    """
    Function to prepare graph data from a CHGCAR file.
    """
    chgcar = Chgcar.from_file(chgcar_file)
    structure = chgcar.structure

    sga = SpacegroupAnalyzer(structure)
    mesh = list(chgcar.data['total'].shape)
    ir_kpoints = sga.get_ir_reciprocal_mesh(mesh)
    irreducible_kpoints = [k[0] for k in ir_kpoints]
    weights = [k[1] for k in ir_kpoints]

    frac_coords = np.array(irreducible_kpoints)
    densities = np.random.rand(len(frac_coords))  # Placeholder for actual density extraction

    avg_properties = calculate_weighted_properties(structure, frac_coords, radius=atom_radius)

    attributes = np.hstack([
        frac_coords,
        np.array(weights).reshape(-1, 1),
        avg_properties
    ])

    neighbors = find_neighbors(frac_coords, radius=kpoint_radius)
    edge_index = []
    for i, kp_neighbors in enumerate(neighbors):
        for neighbor in kp_neighbors:
            edge_index.append([i, neighbor])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x = torch.tensor(attributes, dtype=torch.float)
    y = torch.tensor(densities, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def print_graph(data):
    """
    Function to print the nodes (k-points) with their attributes.
    """
    nodes = data.x.numpy()
    print("Nodes (k-points):")
    for i, node in enumerate(nodes):
        print(f"Node {i}: {node}")


def visualize_graph(data):
    """
    Function to visualize the graph using networkx.
    """
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    nx.draw_networkx(G,
                     pos=nx.spring_layout(G, seed=0),
                     with_labels=False,
                     node_size=100,
                     node_color=data.y,
                     cmap="hsv",
                     vmin=-2,
                     vmax=3,
                     width=0.8,
                     edge_color="grey",
                     font_size=14
                     )
    plt.show()