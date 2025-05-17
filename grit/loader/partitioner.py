import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.distributed.partition import Partitioner, load_partition_info


class PartitionedGraphDataset(InMemoryDataset):
    """
    A PyTorch Geometric Dataset that wraps partitions created by torch_geometric's Partitioner.
    This provides a standard PyG dataset interface to work with partitioned graphs.
    """
    
    def __init__(self, 
                 partition_dir: str, 
                 part_idx: Optional[int] = None,
                 transform=None, 
                 pre_transform=None,
                 pre_filter=None,
                 meta_data: Optional[Dict] = None):
        """
        Initialize a dataset from partitioned graph data.
        
        Args:
            partition_dir: Directory containing the partition files
            part_idx: Specific partition index to load (if None, loads all partitions)
            transform: Transform to apply to the data
            pre_transform: Pre-transform to apply to the data
            pre_filter: Pre-filter to apply to the data
            meta_data: Additional metadata for the dataset
        """
        self.partition_dir = partition_dir
        self.part_idx = part_idx
        self.meta_data = meta_data if meta_data is not None else {}
        
        super(PartitionedGraphDataset, self).__init__(
            root=partition_dir, 
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )
        
        # Load the partitions
        self.data, self.slices = self._load_partitioned_data()
        
    @property
    def raw_file_names(self) -> List[str]:
        """List of raw file names."""
        # This dataset uses pre-processed partition files
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        """List of processed file names."""
        # Return a file that doesn't exist to force processing each time
        # This is because we're working with external partition files
        return ['processed_data.pt']
    
    def download(self):
        """Download the dataset."""
        # No download needed, using existing partition files
        pass
    
    def process(self):
        """Process the dataset."""
        # No processing needed, loading is done in _load_partitioned_data
        pass
    
    def _load_partitioned_data(self) -> Tuple[List[Data], Dict]:
        """
        Load the partitioned data.
        
        Returns:
            Tuple of (data_list, slices)
        """
        # Load partition information
        if self.part_idx is not None:
            # Load specific partition
            partition_data = load_partition_info(self.partition_dir, self.part_idx)
            data_list = [self._convert_partition_to_data(partition_data, self.part_idx)]
        else:
            # Load all partitions
            data_list = []
            # Find the number of partitions by checking directory contents
            partition_files = [f for f in os.listdir(self.partition_dir) 
                              if f.startswith('part_')]
            num_partitions = len(partition_files)
            
            for part_idx in range(num_partitions):
                partition_data = load_partition_info(self.partition_dir, part_idx)
                data = self._convert_partition_to_data(partition_data, part_idx)
                data_list.append(data)
        
        # Create slices for InMemoryDataset format
        slices = self._compute_slices(data_list)
        
        return data_list, slices
    
    def _convert_partition_to_data(self, partition_data: Dict, part_idx: int) -> Data:
        """
        Convert partition information to PyG Data object.
        
        Args:
            partition_data: Dictionary containing partition information
            part_idx: Partition index
            
        Returns:
            PyG Data object
        """
        data = Data()
        
        print(type(partition_data), type(partition_data.keys()[0]))
        # Edge index
        edge_index = self._convert_edge_index(partition_data[4])
        data.edge_index = edge_index
        
        # Node features
        if 'x' in partition_data:
            data.x = partition_data['x']
        
        # Node labels (if available)
        if 'y' in partition_data:
            data.y = partition_data['y']
            
        # Edge features (if available)
        if 'edge_attr' in partition_data:
            data.edge_attr = partition_data['edge_attr']
        
        # Graph-level features (if available)
        if 'u' in partition_data:
            data.u = partition_data['u']
        
        # Add mapping information
        if 'global_node_idx' in partition_data:
            data.global_node_idx = partition_data['global_node_idx']
            
        if 'part_ptr' in partition_data:
            data.part_ptr = partition_data['part_ptr']
            
        # Add partition metadata
        data.part_idx = torch.tensor([part_idx], dtype=torch.long)
        data.num_partitions = torch.tensor([len(os.listdir(self.partition_dir))], dtype=torch.long)
        
        # Add additional metadata from the partition info
        for key, value in partition_data.items():
            if key not in ['edge_index', 'x', 'y', 'edge_attr', 'u', 'global_node_idx', 'part_ptr']:
                setattr(data, key, value)
        
        return data
    
    def _convert_edge_index(self, edge_index_data) -> torch.Tensor:
        """
        Convert edge index from partition format to standard PyG format.
        
        Args:
            edge_index_data: Edge index data from partition
            
        Returns:
            Edge index tensor [2, num_edges]
        """
        if True:
            row, col, _ = edge_index_data.get_all()
            edge_index = torch.stack([row, col], dim=0)
        elif isinstance(edge_index_data, torch.Tensor):
            # Already in the right format
            edge_index = edge_index_data
        else:
            raise TypeError(f"Unexpected edge_index type: {type(edge_index_data)}")
        
        return edge_index
    
    def _compute_slices(self, data_list: List[Data]) -> Dict:
        """
        Compute slices for InMemoryDataset format.
        
        Args:
            data_list: List of Data objects
            
        Returns:
            Dictionary of slices
        """
        # Initialize slices dictionary
        slices = {}
        
        # No data, return empty slices
        if len(data_list) == 0:
            return slices
        
        # For each attribute in the first data object
        for key in data_list[0].keys:
            # Skip non-tensor attributes
            if not hasattr(data_list[0], key) or not isinstance(getattr(data_list[0], key), torch.Tensor):
                continue
                
            # Compute cumulative sizes
            cum_sizes = [0]
            for data in data_list:
                tensor = getattr(data, key)
                cum_sizes.append(cum_sizes[-1] + tensor.size(0))
            
            # Store slices
            slices[key] = torch.tensor(cum_sizes, dtype=torch.long)
        
        return slices
    
    def get_idx_split(self) -> Dict[str, torch.Tensor]:
        """
        Get dataset splits. This mimics the API of PygGraphPropPredDataset.
        In this implementation, all partitions are assigned to the 'train' split.
        
        Returns:
            Dictionary mapping split names to tensors of indices
        """
        num_graphs = len(self)
        indices = list(range(num_graphs))
        
        # If splits are defined in meta_data, use them
        if 'split_idx' in self.meta_data:
            return self.meta_data['split_idx']
        
        # Default: all in train
        return {
            'train': torch.tensor(indices, dtype=torch.long),
            'valid': torch.tensor([], dtype=torch.long),
            'test': torch.tensor([], dtype=torch.long)
        }


def partition_and_convert_to_dataset(
    data: Data,
    num_parts: int,
    partition_dir: str,
    local_node_ratio: float = 0.8,
    meta_data: Optional[Dict] = None
) -> PartitionedGraphDataset:
    """
    Partition a PyG Data object and convert it to a PartitionedGraphDataset.
    
    Args:
        data: PyG Data object to partition
        num_parts: Number of partitions to create
        partition_dir: Directory to save the partitions
        num_workers: Number of workers for parallel processing
        local_node_ratio: Target ratio of nodes that should be local to the partition
        meta_data: Additional metadata for the dataset
        
    Returns:
        PartitionedGraphDataset object
    """
    # Create partition directory if it doesn't exist
    os.makedirs(partition_dir, exist_ok=True)
    
    # Create partitioner
    partitioner = Partitioner(data, num_parts, partition_dir)
    
    # Generate partitions
    partitioner.generate_partition()
    
    # Convert to dataset
    dataset = PartitionedGraphDataset(partition_dir, meta_data=meta_data)
    
    return dataset


def partitioned_dataset_from_existing(
    partition_dir: str,
    meta_data: Optional[Dict] = None
) -> PartitionedGraphDataset:
    """
    Create a PartitionedGraphDataset from existing partition files.
    
    Args:
        partition_dir: Directory containing the partition files
        meta_data: Additional metadata for the dataset
        
    Returns:
        PartitionedGraphDataset object
    """
    return PartitionedGraphDataset(partition_dir, meta_data=meta_data)


# Example usage
def example_partition_ogbn_dataset():
    """Example of partitioning an OGB dataset and converting to PartitionedGraphDataset"""
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
        
        # Load dataset
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/tmp/ogb')
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        
        # Create metadata
        meta_data = {
            'num_classes': dataset.num_classes,
            'split_idx': split_idx,
            'eval_metric': 'accuracy'
        }
        
        # Create partitions
        partition_dir = '/tmp/ogbn-arxiv-partitions'
        num_parts = 4
        
        # Partition and convert
        partitioned_dataset = partition_and_convert_to_dataset(
            data, num_parts, partition_dir, meta_data=meta_data
        )
        
        return partitioned_dataset
        
    except ImportError:
        print("OGB not available. Skipping example.")
        return None


def example_partition_pyg_dataset():
    """Example of partitioning a PyG dataset and converting to PartitionedGraphDataset"""
    # Load a PyG dataset
    dataset = TUDataset(root='/tmp/TUDataset', name='ENZYMES')
    data = dataset[0]  # Get the first graph
    
    # Create metadata
    meta_data = {
        'num_classes': dataset.num_classes,
        'task_type': 'graph_classification'
    }
    
    # Create partitions
    partition_dir = '/tmp/enzymes-partitions'
    num_parts = 2
    
    # Partition and convert
    partitioned_dataset = partition_and_convert_to_dataset(
        data, num_parts, partition_dir, meta_data=meta_data
    )
    
    return partitioned_dataset


def example_load_existing_partitions(partition_dir):
    """Example of loading existing partitions"""
    
    # Check if partitions exist
    if not os.path.exists(partition_dir):
        print(f"Partition directory {partition_dir} does not exist")
        return None
    
    # Load partitions
    dataset = partitioned_dataset_from_existing(partition_dir)
    
    print(f"Loaded dataset with {len(dataset)} partitions")
    for i in range(len(dataset)):
        data = dataset[i]
        print(f"Partition {i}: {data.num_nodes} nodes, {data.num_edges} edges")
    
    return dataset


if __name__ == "__main__":
    print("Running examples...")
    
    # Try PyG dataset example
    try:
        print("\n=== PyG Dataset Example ===")
        partitioned_dataset = example_partition_pyg_dataset()
        if partitioned_dataset:
            print(f"Created dataset with {len(partitioned_dataset)} partitions")
            for i in range(len(partitioned_dataset)):
                data = partitioned_dataset[i]
                print(f"Partition {i}: {data.num_nodes} nodes, {data.num_edges} edges")
    except Exception as e:
        print(f"Error in PyG example: {e}")
    
    # Try OGB dataset example
    try:
        print("\n=== OGB Dataset Example ===")
        partitioned_dataset = example_partition_ogbn_dataset()
        if partitioned_dataset:
            print(f"Created dataset with {len(partitioned_dataset)} partitions")
            for i in range(len(partitioned_dataset)):
                data = partitioned_dataset[i]
                print(f"Partition {i}: {data.num_nodes} nodes, {data.num_edges} edges")
    except Exception as e:
        print(f"Error in OGB example: {e}")
    
    # Try loading existing partitions
    try:
        print("\n=== Load Existing Partitions Example ===")
        example_load_existing_partitions()
    except Exception as e:
        print(f"Error loading existing partitions: {e}")