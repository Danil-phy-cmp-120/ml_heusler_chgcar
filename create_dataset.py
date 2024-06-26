import os
from tqdm import tqdm
from torch_geometric.data import Dataset
from prepare_graph import prepare_data


class CHGCARDataset(Dataset):
    def __init__(self, root_dir, kpoint_radius=0.1, atom_radius=3.0, transform=None, pre_transform=None):
        super(CHGCARDataset, self).__init__(root_dir, transform, pre_transform)
        self.kpoint_radius = kpoint_radius
        self.atom_radius = atom_radius
        self.root_dir = root_dir

    @property
    def raw_file_names(self):
        # Return list of CHGCAR file names in the root_dir
        return os.listdir(self.root_dir)

    def len(self):
        # Return the number of CHGCAR files in the root_dir
        return len(self.raw_file_names)

    def get(self, idx):
        # Read CHGCAR file and process to create a Data object
        chgcar_file = os.path.join(self.root_dir, self.raw_file_names[idx])
        data = prepare_data(chgcar_file, self.kpoint_radius, self.atom_radius)
        return data


if __name__ == "__main__":
    root_dir = "samples_chgcar"

    dataset = CHGCARDataset(root_dir)

    print(f'Graph: {dataset[0]}')
    #print(dataset[0])