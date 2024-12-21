from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict) 
from rdkit import Chem
import numpy as np
from torch_geometric.data import Data
from pprint import pformat
import torch

def graph2smiles(edge_index, edge_attr, x):
    """
    Converts graph data to a SMILES string.
    :param edge_index: Edge indices in COO format (shape [2, num_edges]).
    :param edge_attr: Edge features matrix (shape [num_edges, num_edge_features]).
    :param x: Node feature matrix (shape [num_nodes, num_node_features]).
    :return: SMILES string.
    """
    # Initialize an empty RDKit molecule
    mol = Chem.RWMol()

    # Add atoms to the molecule
    atom_indices = []
    for atom_features in x:
        atomic_num_idx = atom_features[0]  # First feature is atomic number
        atomic_num = allowable_features['possible_atomic_num_list'][atomic_num_idx]
        if atomic_num == 'misc':  # Handle invalid atomic number
            atomic_num = 0  # Assign Hydrogen as a placeholder
        atom_idx = mol.AddAtom(Chem.Atom(atomic_num))
        atom_indices.append(atom_idx)

    # Add bonds to the molecule
    num_edges = edge_index.shape[1]
    for edge_idx in range(num_edges):
        i, j = edge_index[:, edge_idx]  # Get atom indices for the bond
        bond_type_idx = edge_attr[edge_idx, 0]  # First edge feature is bond type
        bond_stereo_idx = edge_attr[edge_idx, 1]  # Second edge feature is bond stereo
        bond_type = allowable_features['possible_bond_type_list'][bond_type_idx]
        bond_stereo = allowable_features['possible_bond_stereo_list'][bond_stereo_idx]
        
        # Handle bond type
        if bond_type == 'SINGLE':
            bond = Chem.BondType.SINGLE
        elif bond_type == 'DOUBLE':
            bond = Chem.BondType.DOUBLE
        elif bond_type == 'TRIPLE':
            bond = Chem.BondType.TRIPLE
        elif bond_type == 'AROMATIC':
            bond = Chem.BondType.AROMATIC
        else:
            continue  # Skip invalid bonds

        # Add the bond
        if not mol.GetBondBetweenAtoms(int(i), int(j)):
            mol.AddBond(int(i), int(j), bond)
            bond_obj = mol.GetBondBetweenAtoms(int(i), int(j))
            
            # Handle bond stereo
            if bond_stereo == 'STEREOZ':
                bond_obj.SetStereo(Chem.BondStereo.STEREOZ)
            elif bond_stereo == 'STEREOE':
                bond_obj.SetStereo(Chem.BondStereo.STEREOE)
            elif bond_stereo == 'STEREOCIS':
                bond_obj.SetStereo(Chem.BondStereo.STEREOCIS)
            elif bond_stereo == 'STEREOTRANS':
                bond_obj.SetStereo(Chem.BondStereo.STEREOTRANS)

    # Generate the SMILES string
    mol.UpdatePropertyCache()
    Chem.SanitizeMol(mol)  # Ensure the molecule is valid
    smiles = Chem.MolToSmiles(mol, canonical=True)

    return smiles







def graph2token2input(
    data, gtokenizer
):
    ls_embed = []
    if isinstance(data, Data):
        graph = data
        print(f"Inspecting tokenization results!\nTokenize graph:\n{data}")
        token_res = gtokenizer.tokenize(graph)
        # print(
        #     f"\nTokens:\n{pformat(token_res.ls_tokens)}\nLabels:\n{pformat(token_res.ls_labels)}\nembed:{np.array(token_res.ls_embed)}\n"
        # )
        tokens, labels, ls_embed, ls_len = (
            gtokenizer.pack_token_seq(token_res, idx)
            if gtokenizer.mpe is not None
            else (
                token_res.ls_tokens,
                token_res.ls_labels,
                token_res.ls_embed,
                [len(token_res.ls_tokens)],
            )
        )
        print(
            f"Packed Tokens:\n{pformat(tokens)}\nPacked Labels:\n{pformat(labels)}\nPacked embed:\n{np.array(ls_embed).shape}\n{np.array(ls_embed)}\nPacked len:\n{pformat(ls_len)}"
        ) if gtokenizer.mpe is not None else None
        in_dict = gtokenizer.convert_tokens_to_ids(tokens, labels)
        if ls_embed:  # for pretty print purpose ONLY
            in_dict["embed"] = np.array(ls_embed)
        # print(f"Tokenized results:\n{pformat(in_dict)}\n")
        if ls_embed:
            in_dict["embed"] = ls_embed
        token_res.ls_tokens = tokens
        token_res.ls_labels = labels
        token_res.ls_embed = ls_embed
        token_res.ls_len = ls_len
        inputs = gtokenizer.prepare_inputs_for_task(
            in_dict,
            graph,
            token_res=token_res,
        )
    elif isinstance(data, Dict):
        inputs = data
    else:
        raise ValueError(f"Type {type(data)} of data {data} is NOT implemented yet!")
    if ls_embed:  # for pretty print purpose ONLY
        inputs["embed"] = np.array(ls_embed)
    # print(f"Inputs for model:\n{pformat(inputs)}\n")
    gtokenizer.set_eos_idx(inputs["input_ids"])
    if token_res.ls_embed is not None:
        return token_res.ls_tokens, token_res.ls_labels, token_res.ls_embed, inputs
    else:
        return token_res.ls_tokens, token_res.ls_labels, [], inputs
    





def convert_to_tensors(inputs):
    """
    Convert all values in a dictionary to PyTorch tensors.
    
    Args:
        inputs (dict): A dictionary containing keys and values to be converted.
        
    Returns:
        dict: A dictionary with the same keys but values as PyTorch tensors.
    """
    tensor_dict = {}
    for key, value in inputs.items():
        # Convert to tensor if not already a tensor
        tensor = torch.tensor(value) if not isinstance(value, torch.Tensor) else value
        # Add batch size of 1 to the first dimension
        tensor_dict[key] = tensor.unsqueeze(0)
    return tensor_dict



def ReorderCanonicalRankAtoms(mol):
    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum, order

def smiles2graph(smiles_string, removeHs=True, reorder_atoms=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    mol = mol if removeHs else Chem.AddHs(mol)
    if reorder_atoms:
        mol, _ = ReorderCanonicalRankAtoms(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_attr'] = edge_attr
    graph['x'] = x
    graph['num_nodes'] = len(x)

    return graph 


def graph_to_torch_geometric(graph):
    """
    Converts a dictionary representation of a graph to a PyTorch Geometric Data object.
    :param graph: Dictionary containing graph data with keys:
                  - 'edge_index': Connectivity matrix (array of shape [2, num_edges])
                  - 'edge_attr': Edge features (array of shape [num_edges, num_edge_features])
                  - 'node_feat': Node features (array of shape [num_nodes, num_node_features])
                  - 'num_nodes': Number of nodes (int)
    :return: PyTorch Geometric Data object
    """
    edge_index = torch.tensor(graph['edge_index'], dtype=torch.int64)  # Edge indices
    edge_attr = torch.tensor(graph['edge_attr'], dtype=torch.int64)  # Edge attributes
    x = torch.tensor(graph['x'], dtype=torch.int64)          # Node features
    num_nodes = graph.get('num_nodes', x.size(0))                    # Number of nodes

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    return data


# Step 1: Load the vocabulary
def load_vocab(vocab_file):
    """Loads a vocabulary file and returns a dictionary mapping token IDs to tokens."""
    vocab = {}
    with open(vocab_file, "r") as f:
        for line in f:
            value, key = line.split()
            vocab[int(key)] = value  # Remove newline characters
    return vocab

# Step 2: Convert predicted label IDs to tokens
def convert_labels_to_tokens(labels, vocab):
    """Converts label IDs to tokens using the provided vocabulary."""
    tokens = [vocab[label.item()] for label in labels.view(-1)]
    return tokens




def token_to_graph(linearized):
    """
    Convert a linearized graph representation back to graph format.
    
    Args:
        linearized: List of lists representing the linearized graph.

    Returns:
        graph_x: Array of node features.
        graph_edge_index: Array of edge indices.
        graph_edge_attr: Array of edge attributes.
    """
    node_features = []
    edge_indices = []
    edge_attributes = []
    
    node_mapping = {}
    edge_set = set()
    
    for i, sub_list in enumerate(linearized):
        # Extract node index and features
        node_idx = int(sub_list[0])
        
        if node_idx not in node_mapping:
            node_mapping[node_idx] = len(node_features)
            node_feats = [int(feature.split("#")[-1]) for feature in sub_list[1:10]]
            node_features.append(node_feats)
        
        # Extract edge features, skip the first node's dummy edge
        if i > 0:
            edge_feats = [int(feature.split("#")[-1]) for feature in sub_list[10:13]]
            prev_node_idx = int(linearized[i - 1][0])
            edge = (node_mapping[prev_node_idx], node_mapping[node_idx])
            
            if edge not in edge_set:
                edge_set.add(edge)
                edge_set.add(edge[::-1])  # Add the inverse edge
                edge_attributes.append(edge_feats)
                edge_indices.append([edge[0], edge[1]])
                edge_attributes.append(edge_feats)  # Duplicate edge attributes for inverse edge
                edge_indices.append([edge[1], edge[0]])

    # Convert lists to arrays
    graph_x = np.array(node_features, dtype=int)
    graph_edge_index = np.array(edge_indices, dtype=int).T
    graph_edge_attr = np.array(edge_attributes, dtype=int)

    return graph_x, graph_edge_index, graph_edge_attr



def pad_to_max_length(arr, max_length, pad_value=0):
    """
    Pads a numpy array to the specified max_length with pad_value.
    
    Args:
        arr: Input numpy array of shape (N, D) or (N,).
        max_length: Desired maximum length.
        pad_value: Value to pad with.

    Returns:
        Padded numpy array of shape (max_length, D) or (max_length,).
    """
    if len(arr.shape) == 1:  # Handle 1D arrays
        if arr.shape[0] > max_length:
            raise ValueError(f"Input array length {arr.shape[0]} exceeds maximum allowed length {max_length}.")
        padding = np.full((max_length - arr.shape[0],), pad_value)
        return np.concatenate([arr, padding])
    elif len(arr.shape) == 2:  # Handle 2D arrays
        if arr.shape[0] > max_length:
            raise ValueError(f"Input array length {arr.shape[0]} exceeds maximum allowed length {max_length}.")
        padding = np.full((max_length - arr.shape[0], arr.shape[1]), pad_value)
        return np.vstack([arr, padding])
    else:
        raise ValueError("Only 1D and 2D arrays are supported for padding.")

def graph2token2input_generation(
    data, gtokenizer, num_input, max_length
):
    ls_embed = []
    if isinstance(data, Data):
        graph = data
        print(f"Inspecting tokenization results!\nTokenize graph:\n{data}")
        token_res = gtokenizer.tokenize(graph)
        # print(
        #     f"\nTokens:\n{pformat(token_res.ls_tokens)}\nLabels:\n{pformat(token_res.ls_labels)}\nembed:{np.array(token_res.ls_embed)}\n"
        # )
        tokens, labels, ls_embed, ls_len = (
            gtokenizer.pack_token_seq(token_res, idx)
            if gtokenizer.mpe is not None
            else (
                token_res.ls_tokens,
                token_res.ls_labels,
                token_res.ls_embed,
                [len(token_res.ls_tokens)],
            )
        )
        if gtokenizer.mpe is not None:
            print(
                f"Packed Tokens:\n{tokens}\nPacked Labels:\n{labels}\nPacked embed:\n{np.array(ls_embed).shape}\n{np.array(ls_embed)}\nPacked len:\n{ls_len}"
            )

        label_tokens = tokens

        in_dict = gtokenizer.convert_tokens_to_ids(tokens, labels)


        # Truncate to num_input rows
        tokens = tokens[:num_input]
        # labels = labels[:num_input]
        # tokens = pad_to_max_length(np.array(tokens), max_length, pad_value=0)
        # labels = pad_to_max_length(np.array(labels), max_length, pad_value=-100)  # Use -100 for ignored labels
        if ls_embed:
            ls_embed = ls_embed[:num_input]
            # ls_embed = pad_to_max_length(np.array(ls_embed), max_length, pad_value=0.0)


        if ls_embed:
            in_dict["embed"] = ls_embed

        # Update in_dict with padded values
        # in_dict["input_ids"] = pad_to_max_length(np.array(in_dict["input_ids"][:num_input]), max_length, pad_value=0)
        # in_dict["position_ids"] = np.arange(max_length)
        # in_dict["labels"] = pad_to_max_length(np.array(in_dict["labels"]), max_length, pad_value=-100)
        # in_dict["attention_mask"] = pad_to_max_length(np.ones((num_input,)), max_length, pad_value=0)
        in_dict["input_ids"] = in_dict["input_ids"][:num_input]
        in_dict["position_ids"] = np.arange(num_input)
        # in_dict["labels"] = pad_to_max_length(np.array(in_dict["labels"]), max_length, pad_value=-100)
        in_dict["attention_mask"] = np.ones((num_input,))

        
        token_res.ls_tokens = tokens
        token_res.ls_labels = labels
        token_res.ls_embed = ls_embed
        token_res.ls_len = ls_len
        inputs = gtokenizer.prepare_inputs_for_task(
            in_dict,
            graph,
            token_res=token_res,
        )
    elif isinstance(data, Dict):
        inputs = data
    else:
        raise ValueError(f"Type {type(data)} of data {data} is NOT implemented yet!")
    if ls_embed:  # for pretty print purpose ONLY
        inputs["embed"] = np.array(ls_embed)
    # print(f"Inputs for model:\n{pformat(inputs)}\n")
    gtokenizer.set_eos_idx(inputs["input_ids"])
    if token_res.ls_embed is not None:
        return token_res.ls_tokens, token_res.ls_labels, token_res.ls_embed, inputs
    else:
        return token_res.ls_tokens, token_res.ls_labels, [], inputs