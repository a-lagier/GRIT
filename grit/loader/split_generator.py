import json
import logging
import os

import random
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, download_url
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import index2mask, set_dataset_attr

def prepare_splits(dataset):
    """Ready train/val/test splits.

    Determine the type of split from the config and call the corresponding
    split generation / verification function.
    """
    split_mode = cfg.dataset.split_mode

    if split_mode == 'standard':
        setup_standard_split(dataset)
    elif split_mode == 'random':
        setup_random_split(dataset)
    elif split_mode.startswith('cv-'):
        cv_type, k = split_mode.split('-')[1:]
        setup_cv_split(dataset, cv_type, int(k))
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")


def setup_standard_split(dataset):
    """Select a standard split.

    Use standard splits that come with the dataset. Pick one split based on the
    ``split_index`` from the config file if multiple splits are available.

    GNNBenchmarkDatasets have splits that are not prespecified as masks. Therefore,
    they are handled differently and are first processed to generate the masks.

    Raises:
        ValueError: If any one of train/val/test mask is missing.
        IndexError: If the ``split_index`` is greater or equal to the total
            number of splits available.
    """
    split_index = cfg.dataset.split_index
    task_level = cfg.dataset.task

    if task_level == 'node':
        for split_name in 'train_mask', 'val_mask', 'test_mask':
            mask = getattr(dataset.data, split_name, None)
            # Check if the train/val/test split mask is available
            if mask is None:
                raise ValueError(f"Missing '{split_name}' for standard split")

            # Pick a specific split if multiple splits are available
            if mask.dim() == 2:
                if split_index >= mask.shape[1]:
                    raise IndexError(f"Specified split index ({split_index}) is "
                                     f"out of range of the number of available "
                                     f"splits ({mask.shape[1]}) for {split_name}")
                set_dataset_attr(dataset, split_name, mask[:, split_index],
                                 len(mask[:, split_index]))
            else:
                if split_index != 0:
                    raise IndexError(f"This dataset has single standard split")

    elif task_level == 'graph':
        for split_name in 'train_graph_index', 'val_graph_index', 'test_graph_index':
            if not hasattr(dataset.data, split_name):
                raise ValueError(f"Missing '{split_name}' for standard split")
        if split_index != 0:
            raise NotImplementedError(f"Multiple standard splits not supported "
                                      f"for dataset task level: {task_level}")

    elif task_level == 'link_pred':
        for split_name in 'train_edge_index', 'val_edge_index', 'test_edge_index':
            if not hasattr(dataset.data, split_name):
                raise ValueError(f"Missing '{split_name}' for standard split")
        if split_index != 0:
            raise NotImplementedError(f"Multiple standard splits not supported "
                                      f"for dataset task level: {task_level}")

    else:
        if split_index != 0:
            raise NotImplementedError(f"Multiple standard splits not supported "
                                      f"for dataset task level: {task_level}")


def setup_random_split(dataset):
    """Generate random splits.

    Generate random train/val/test based on the ratios defined in the config
    file.

    Raises:
        ValueError: If the number split ratios is not equal to 3, or the ratios
            do not sum up to 1.
    """
    split_ratios = cfg.dataset.split

    if len(split_ratios) != 3:
        raise ValueError(
            f"Three split ratios is expected for train/val/test, received "
            f"{len(split_ratios)} split ratios: {repr(split_ratios)}")
    elif sum(split_ratios) != 1:
        raise ValueError(
            f"The train/val/test split ratios must sum up to 1, input ratios "
            f"sum up to {sum(split_ratios):.2f} instead: {repr(split_ratios)}")

    train_index, val_test_index = next(
        ShuffleSplit(
            train_size=split_ratios[0],
            random_state=cfg.seed
        ).split(dataset.data.y, dataset.data.y)
    )
    val_index, test_index = next(
        ShuffleSplit(
            train_size=split_ratios[1] / (1 - split_ratios[0]),
            random_state=cfg.seed
        ).split(dataset.data.y[val_test_index], dataset.data.y[val_test_index])
    )
    val_index = val_test_index[val_index]
    test_index = val_test_index[test_index]

    set_dataset_splits(dataset, [train_index, val_index, test_index])


def set_dataset_splits(dataset, splits):
    """Set given splits to the dataset object.

    Args:
        dataset: PyG dataset object
        splits: List of train/val/test split indices

    Raises:
        ValueError: If any pair of splits has intersecting indices
    """
    # First check whether splits intersect and raise error if so.
    for i in range(len(splits) - 1):
        for j in range(i + 1, len(splits)):
            n_intersect = len(set(splits[i]) & set(splits[j]))
            if n_intersect != 0:
                raise ValueError(
                    f"Splits must not have intersecting indices: "
                    f"split #{i} (n = {len(splits[i])}) and "
                    f"split #{j} (n = {len(splits[j])}) have "
                    f"{n_intersect} intersecting indices"
                )

    task_level = cfg.dataset.task
    if task_level == 'node':
        split_names = ['train_mask', 'val_mask', 'test_mask']
        for split_name, split_index in zip(split_names, splits):
            mask = index2mask(split_index, size=dataset.data.y.shape[0])
            set_dataset_attr(dataset, split_name, mask, len(mask))

    elif task_level == 'graph':
        split_names = [
            'train_graph_index', 'val_graph_index', 'test_graph_index'
        ]
        for split_name, split_index in zip(split_names, splits):
            set_dataset_attr(dataset, split_name, split_index, len(split_index))

    else:
        raise ValueError(f"Unsupported dataset task level: {task_level}")


def setup_cv_split(dataset, cv_type, k):
    """Generate cross-validation splits.

    Generate `k` folds for cross-validation based on `cv_type` procedure. Save
    these to disk or load existing splits, then select particular train/val/test
    split based on cfg.dataset.split_index from the config object.

    Args:
        dataset: PyG dataset object
        cv_type: Identifier for which sklearn fold splitter to use
        k: how many cross-validation folds to split the dataset into

    Raises:
        IndexError: If the `split_index` is greater than or equal to `k`
    """
    split_index = cfg.dataset.split_index
    split_dir = cfg.dataset.split_dir

    if split_index >= k:
        raise IndexError(f"Specified split_index={split_index} is "
                         f"out of range of the number of folds k={k}")

    os.makedirs(split_dir, exist_ok=True)
    save_file = os.path.join(
        split_dir,
        f"{cfg.dataset.format}_{dataset.name}_{cv_type}-{k}.json"
    )
    if not os.path.isfile(save_file):
        create_cv_splits(dataset, cv_type, k, save_file)
    with open(save_file) as f:
        cv = json.load(f)
    assert cv['dataset'] == dataset.name, "Unexpected dataset CV splits"
    assert cv['n_samples'] == len(dataset), "Dataset length does not match"
    assert cv['n_splits'] > split_index, "Fold selection out of range"
    assert k == cv['n_splits'], f"Expected k={k}, but {cv['n_splits']} found"

    test_ids = cv[str(split_index)]
    val_ids = cv[str((split_index + 1) % k)]
    train_ids = []
    for i in range(k):
        if i != split_index and i != (split_index + 1) % k:
            train_ids.extend(cv[str(i)])

    set_dataset_splits(dataset, [train_ids, val_ids, test_ids])


def create_cv_splits(dataset, cv_type, k, file_name):
    """Create cross-validation splits and save them to file.
    """
    n_samples = len(dataset)
    if cv_type == 'stratifiedkfold':
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=123)
        kf_split = kf.split(np.zeros(n_samples), dataset.data.y)
    elif cv_type == 'kfold':
        kf = KFold(n_splits=k, shuffle=True, random_state=123)
        kf_split = kf.split(np.zeros(n_samples))
    else:
        ValueError(f"Unexpected cross-validation type: {cv_type}")

    splits = {'n_samples': n_samples,
              'n_splits': k,
              'cross_validator': kf.__str__(),
              'dataset': dataset.name
              }
    for i, (_, ids) in enumerate(kf_split):
        splits[i] = ids.tolist()
    with open(file_name, 'w') as f:
        json.dump(splits, f)
    logging.info(f"[*] Saved newly generated CV splits by {kf} to {file_name}")













class PartitionedGraphDataset(InMemoryDataset):
    """
    A class that mimics the interface of PygGraphPropPredDataset for partitioned graphs.
    This class holds the partitioned subgraphs and provides an interface similar to 
    PyTorch Geometric datasets.
    """
    
    def __init__(self, subgraphs, name = "partitioned_dataset"):
        """
        Initialize a new partitioned graph dataset.
        
        Args:
            subgraphs: Dictionary mapping partition IDs to PyTorch Geometric Data objects
            name: Name of the dataset
        """
        self.subgraphs = subgraphs
        self.name = name
        self.partition_ids = sorted(list(subgraphs.keys()))
        self._indices = list(range(len(self.partition_ids)))
        
        # Extract metadata from subgraphs
        sample_graph = list(subgraphs.values())[0]
        self.num_features = sample_graph.x.shape[1] if hasattr(sample_graph, 'x') and sample_graph.x is not None else 0
        self.num_classes = 0
        
        if hasattr(sample_graph, 'y') and sample_graph.y is not None:
            if sample_graph.y.dim() > 0:
                self.num_classes = sample_graph.y.shape[1] if sample_graph.y.dim() > 1 else 1
        
        # Create a task type (node or graph prediction)
        sample_y = sample_graph.y if hasattr(sample_graph, 'y') else None
        if sample_y is not None:
            if sample_y.shape[0] == sample_graph.num_nodes:
                self.task_type = 'node'
            else:
                self.task_type = 'graph'
        else:
            self.task_type = 'unknown'
    
    def __len__(self):
        """Return the number of partitions."""
        return len(self.partition_ids)
    
    def __getitem__(self, idx):
        """Get a specific partition by index."""
        if isinstance(idx, int):
            partition_id = self.partition_ids[idx]
            return self.subgraphs[partition_id]
        elif isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self[i] for i in indices]
        elif isinstance(idx, list):
            return [self[i] for i in idx]
        else:
            raise TypeError("Invalid index type")
    
    def get(self, idx):
        return self.__getitem__(idx)
    
    def get_idx_split(self):
        """
        Get dataset splits. This mimics the API of PygGraphPropPredDataset.
        In this implementation, all partitions are assigned to the 'train' split.
        
        Returns:
            Dictionary mapping split names to tensors of indices
        """
        return {
            'train': torch.tensor(self._indices, dtype=torch.long),
            'valid': torch.tensor([], dtype=torch.long),
            'test': torch.tensor([], dtype=torch.long)
        }
    
    def get_partition(self, partition_id: int):
        """Get a specific partition by its ID."""
        return self.subgraphs[partition_id]
    
    def to(self, device):
        """Move all subgraphs to the specified device."""
        for partition_id in self.partition_ids:
            self.subgraphs[partition_id] = self.subgraphs[partition_id].to(device)
        return self








def create_subgraphs(data, partition_dict):
    """
    Create subgraphs for each partition.
    
    Args:
        data: PyTorch Geometric Data object
        partition_dict: Dictionary mapping partition IDs to lists of node indices
        
    Returns:
        Dictionary mapping partition IDs to PyTorch Geometric Data objects
    """
    subgraphs = {}
    
    for part, nodes in partition_dict.items():
        # Create node mapping from original to new indices
        node_idx = torch.tensor(nodes, dtype=torch.long)
        node_mapping = {int(old_idx): new_idx for new_idx, old_idx in enumerate(node_idx)}
        
        # Filter edges within this partition
        mask1 = torch.isin(data.edge_index[0], node_idx)
        mask2 = torch.isin(data.edge_index[1], node_idx)
        mask = mask1 & mask2
        edge_index = data.edge_index[:, mask]
        
        # Remap node indices
        new_edge_index = torch.zeros_like(edge_index)
        for i in range(edge_index.shape[1]):
            new_edge_index[0, i] = node_mapping[int(edge_index[0, i])]
            new_edge_index[1, i] = node_mapping[int(edge_index[1, i])]
        
        # Filter and remap node features
        x = data.x[node_idx] if hasattr(data, 'x') and data.x is not None else None
        
        # Create new Data object
        subgraph = Data(x=x, edge_index=new_edge_index)
        
        # Add other attributes if they exist
        if hasattr(data, 'y') and data.y is not None:
            if data.y.shape[0] == data.num_nodes:  # Node-level targets
                subgraph.y = data.y[node_idx]
            else:  # Graph-level targets
                subgraph.y = data.y
                
        # Add original node indices for reference
        subgraph.original_indices = node_idx
        
        subgraphs[part] = subgraph
    
    return PartitionedGraphDataset(subgraphs)

def compute_partition_metrics(data, partition_dict):
    """
    Compute metrics to evaluate the quality of the partitioning.
    
    Args:
        data: PyTorch Geometric Data object
        partition_dict: Dictionary mapping partition IDs to lists of node indices
        
    Returns:
        Dictionary of metrics
    """
    edge_index = data.edge_index
    num_edges = edge_index.shape[1]
    num_nodes = data.num_nodes
    
    # Calculate edge cut
    edge_cut = 0
    
    # Create a mapping from node to partition
    node_to_part = {}
    for part, nodes in partition_dict.items():
        for node in nodes:
            node_to_part[node] = part
    
    # Count edges between different partitions
    for i in range(num_edges):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if node_to_part.get(src) != node_to_part.get(dst):
            edge_cut += 1
    
    # Calculate partition size statistics
    part_sizes = [len(nodes) for nodes in partition_dict.values()]
    avg_size = sum(part_sizes) / len(part_sizes)
    max_size = max(part_sizes)
    min_size = min(part_sizes)
    size_imbalance = max_size / avg_size
    
    metrics = {
        'edge_cut': edge_cut,
        'edge_cut_ratio': edge_cut / num_edges,
        'avg_partition_size': avg_size,
        'max_partition_size': max_size,
        'min_partition_size': min_size,
        'size_imbalance': size_imbalance
    }
    
    return metrics


def random_partition(data, num_parts):
    """
    Partition a PyTorch Geometric graph randomly.
    
    Args:
        data: PyTorch Geometric Data object
        num_parts: Number of partitions to create
        
    Returns:
        Dictionary mapping partition IDs to lists of node indices
    """
    n_nodes = data.num_nodes
    nodes = list(range(n_nodes))
    random.shuffle(nodes)
    
    partition_dict = {}
    for i, node in enumerate(nodes):
        part = i % num_parts
        if part not in partition_dict:
            partition_dict[part] = []
        partition_dict[part].append(node)
    
    return partition_dict

def edge_partition(data, num_parts, max_iter=10):
    """
    Partition a PyTorch Geometric graph using a simple edge-cut heuristic.
    
    Args:
        data: PyTorch Geometric Data object
        num_parts: Number of partitions to create
        max_iter: Maximum number of iterations for refinement
        
    Returns:
        Dictionary mapping partition IDs to lists of node indices
    """
    # Start with a random partition
    partition_dict = random_partition(data, num_parts)
    edge_index = data.edge_index
    
    # Mapping from node to partition
    node_to_part = {}
    for part, nodes in partition_dict.items():
        for node in nodes:
            node_to_part[node] = part
    
    # Iteratively refine the partition
    for _ in range(max_iter):
        changes_made = 0
        
        # For each node, check if moving it to another partition would reduce edge cuts
        for node in tqdm(range(data.num_nodes)):
            current_part = node_to_part[node]
            
            # Find all neighbors
            mask = (edge_index[0] == node) | (edge_index[1] == node)
            neighbors1 = edge_index[1][edge_index[0] == node]
            neighbors2 = edge_index[0][edge_index[1] == node]
            neighbors = torch.cat([neighbors1, neighbors2]).tolist()
            
            # Count neighbors in each partition
            part_counts = {}
            for neighbor in neighbors:
                neighbor_part = node_to_part.get(neighbor, current_part)
                part_counts[neighbor_part] = part_counts.get(neighbor_part, 0) + 1
            
            # Find best partition (with most neighbors)
            best_part = current_part
            best_count = part_counts.get(current_part, 0)
            
            for part, count in part_counts.items():
                if count > best_count:
                    best_part = part
                    best_count = count
            
            # Move node if beneficial
            if best_part != current_part:
                partition_dict[current_part].remove(node)
                partition_dict[best_part].append(node)
                node_to_part[node] = best_part
                changes_made += 1
        
        # Stop if no improvements
        if changes_made == 0:
            break
    
    return partition_dict

def split_graph(dataset, num_parts=10, partition_method='edge'):
    if partition_method == 'edge':
        partition = edge_partition(dataset, num_parts)
    else:
        partition = random_partition(dataset, num_parts)

    print(compute_partition_metrics(dataset, partition))

    pyg_dataset = create_subgraphs(dataset, partition)
