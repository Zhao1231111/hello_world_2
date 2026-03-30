import os
import glob
import torch
from torch.utils.data import IterableDataset, DataLoader
import random
import math

class StreamingShardedDataset(IterableDataset):
    """
    An IterableDataset that loads `.pt` shard files (buckets) dynamically,
    maintains a shuffle buffer, and yields exactly (batch_size) samples.
    """
    def __init__(self, shards_dir: str, buffer_size_mb: int = 512, max_buffer_points: int = 300000):
        super().__init__()
        self.shards_dir = shards_dir
        self.shard_files = sorted(glob.glob(os.path.join(shards_dir, "*.pt")))
        if not self.shard_files:
            raise FileNotFoundError(f"No .pt shard chunks found in {shards_dir}.")
            
        print(f"[StreamingDataset] Found {len(self.shard_files)} shards.")
        self.buffer_size_mb = buffer_size_mb
        self.max_buffer_points = max_buffer_points

    def _estimate_effective_max_points(self, x_tensor, y_tensor):
        """
        根据单点字节开销估算“按内存上限可容纳的最大点数”。
        最终上限 = min(max_buffer_points, 按 MB 预算估算的点数)。
        """
        if self.buffer_size_mb <= 0:
            return max(1, self.max_buffer_points)

        x_bytes = x_tensor.element_size() * x_tensor.shape[1]
        y_bytes = y_tensor.element_size() * y_tensor.shape[1]
        bytes_per_point = x_bytes + y_bytes
        if bytes_per_point <= 0:
            return max(1, self.max_buffer_points)

        max_points_by_mb = int((self.buffer_size_mb * 1024 * 1024) // bytes_per_point)
        if max_points_by_mb <= 0:
            max_points_by_mb = 1
        return max(1, min(self.max_buffer_points, max_points_by_mb))
        
    def _load_shard(self, path):
        # returns dict with 'X' and 'Y'
        return torch.load(path, map_location='cpu', weights_only=False)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process data loading
            file_queue = self.shard_files.copy()
        else:
            # Multi-process data loading. Split the shard files so workers don't read the same buckets simultaneously.
            per_worker = int(math.ceil(len(self.shard_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            file_queue = self.shard_files[worker_id * per_worker:(worker_id + 1) * per_worker]
            
        random.shuffle(file_queue)
        
        buffer_X = None
        buffer_Y = None
        effective_max_points = None
        
        while True:
            # Refill buffer if items drop below threshold and we still have files
            refill_threshold = self.max_buffer_points // 2
            if effective_max_points is not None:
                refill_threshold = effective_max_points // 2
            refill_threshold = max(1, refill_threshold)

            while (buffer_X is None or buffer_X.shape[0] < refill_threshold) and file_queue:
                next_file = file_queue.pop(0)
                try:
                    data = self._load_shard(next_file)
                    new_X = data['X']
                    new_Y = data['Y']

                    # 第一次读到 shard 后，基于特征维度估算有效缓冲上限。
                    if effective_max_points is None:
                        effective_max_points = self._estimate_effective_max_points(new_X, new_Y)
                        print(
                            "[StreamingDataset] Buffer limit = "
                            f"{effective_max_points} points "
                            f"(budget={self.buffer_size_mb}MB, hard_cap={self.max_buffer_points})."
                        )
                    
                    if buffer_X is None:
                        buffer_X, buffer_Y = new_X, new_Y
                    else:
                        buffer_X = torch.cat([buffer_X, new_X], dim=0)
                        buffer_Y = torch.cat([buffer_Y, new_Y], dim=0)
                        
                    # Mixed shuffle deeply
                    N = buffer_X.shape[0]
                    perm = torch.randperm(N)
                    buffer_X = buffer_X[perm]
                    buffer_Y = buffer_Y[perm]
                    
                except Exception as e:
                    print(f"Failed to read shard {next_file}: {e}")
                    
            # Check if completely out of data
            if buffer_X is None or buffer_X.shape[0] == 0:
                break
                
            # Pop ONE sample from the end of the buffer tensor. 
            # Doing this continuously will eventually drain buffer_X until the while-loop refills it.
            x_sample = buffer_X[-1]
            y_sample = buffer_Y[-1]
            
            buffer_X = buffer_X[:-1]
            buffer_Y = buffer_Y[:-1]
            
            yield x_sample, y_sample


def get_sharded_dataloader(
    shards_dir: str,
    batch_size: int = 8192,
    num_workers: int = 2,
    buffer_size_mb: int = 512,
    max_buffer_points: int = 300000,
):
    """
    Initialize the DataLoader on top of the iterable dataset.
    Note: if num_workers > 0, each worker gets a FULL copy of the iterable dataset.
    To avoid overlapping identical files, you'd need worker_init_fn to shard the file_queue.
    For simplicity and raw GPU saturation, num_workers=0 or 1 is often sufficient
    if the buffer holds millions of points.
    """
    dataset = StreamingShardedDataset(
        shards_dir,
        buffer_size_mb=buffer_size_mb,
        max_buffer_points=max_buffer_points,
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        # shuffle=True is implicitly handled by the Iterable buffer
        num_workers=num_workers,
        pin_memory=True,     
        drop_last=False
    )
    return dataloader

# You can now import `get_sharded_dataloader` in your training scripts.
# See train_regressor.py for usage.
