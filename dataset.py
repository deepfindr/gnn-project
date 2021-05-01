#%% Load Dataframe
import pandas as pd
DATA_PATH = "data/HIV.csv"
data = pd.read_csv(DATA_PATH)
data.head()

#%% Show sample molecules
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw

sample_smiles = data["smiles"][4:30].values
sample_mols = [Chem.MolFromSmiles(smiles) for \
                smiles in sample_smiles]
grid = Draw.MolsToGridImage(sample_mols,
                    molsPerRow=4,
                    subImgSize=(200,200))
grid

#%% Quick check with versions
import torch
print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")

#%% Generate a dataset
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from rdkit.Chem import rdmolops
from scipy import sparse

class MoleculeDataset(Dataset):
    def __init__(self, data_path):
        self.processed_dir = "data/processed/"
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)

    def process(self):
        for index, mol in self.data.iterrows():
            mol = Chem.MolFromSmiles(mol["smiles"])
            # Get node features
            node_feats = self._get_node_features(mol)
            # Get edge features
            edge_feats = self._get_edge_features(mol)
            # Get adjacency info
            edge_index = self._get_adjacency_info(mol)

            # Create data object
            data = Data(x=node_feats, 
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=[mol["HIV_active"]], # Graph-level labels
                        smiles=mol["smiles"]
                        ) 
            torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))


    def _get_node_features(self, mol):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number        
            node_feats.append(atom.GetAtomicNum())
            # Feature 2: Atom degree
            node_feats.append(atom.GetDegree())
            # Feature 3: Formal charge
            node_feats.append(atom.GetFormalCharge())
            # Feature 4: Hybridization
            node_feats.append(atom.GetHybridization())
            # Feature 5: Aromaticity
            node_feats.append(atom.GetIsAromatic())
            # Append node features to matrix
            all_node_feats.append(node_feats)
        
        return np.asarray(all_node_feats)


    def _get_edge_features(self, mol):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix
            all_edge_feats.append(edge_feats)

        return np.asarray(all_edge_feats)


    def _get_adjacency_info(self, mol):
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        row, col = np.where(adj_matrix)
        coo = np.array(list(zip(row, col)))
        coo = np.reshape(coo, (2, -1))
        return coo


    def len(self):
        return len(self.data.shape[0])

    def get(self, idx):
        """ Equivalent to __getitem__ in pytorch """
        data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))
        return data




#%% Test the dataset
dataset = MoleculeDataset(data_path="data/HIV.csv")
dataset[0]