import json
import math
import os

import numpy as np
import torch
from plyfile import PlyData
from plyfile import PlyElement
from tqdm import tqdm


def load_ply_gt(ply_path):
    """
    读取 point_cloud.ply，构建 GT 张量字典和 id->行号映射。
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"GT PLY file not found: {ply_path}")

    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    ids = np.asarray(vertex["id"])

    f_dc_names = [p.name for p in vertex.properties if p.name.startswith("f_dc_")]
    f_rest_names = [p.name for p in vertex.properties if p.name.startswith("f_rest_")]
    scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
    rot_names = [p.name for p in vertex.properties if p.name.startswith("rot_")]

    f_dc_names.sort(key=lambda x: int(x.split("_")[-1]))
    f_rest_names.sort(key=lambda x: int(x.split("_")[-1]))
    scale_names.sort(key=lambda x: int(x.split("_")[-1]))
    rot_names.sort(key=lambda x: int(x.split("_")[-1]))

    gt_dict = {
        "xyz": torch.from_numpy(np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)).float(),
        "f_dc": torch.from_numpy(np.stack([vertex[name] for name in f_dc_names], axis=-1)).float(),
        "f_rest": torch.from_numpy(np.stack([vertex[name] for name in f_rest_names], axis=-1)).float(),
        "opacity": torch.sigmoid(torch.from_numpy(vertex["opacity"]).float().unsqueeze(1)),
        "scale": torch.from_numpy(np.stack([vertex[name] for name in scale_names], axis=-1)).float(),
        "rot": torch.from_numpy(np.stack([vertex[name] for name in rot_names], axis=-1)).float(),
    }

    # 数据清洗：强制 sx >= sy，减少参数对称歧义。
    scale_tensor = gt_dict["scale"]
    rot_tensor = gt_dict["rot"]
    mask = scale_tensor[:, 0] < scale_tensor[:, 1]

    raw_vertex_data = vertex.data.copy()

    if mask.any():
        temp = scale_tensor[mask, 0].clone()
        scale_tensor[mask, 0] = scale_tensor[mask, 1]
        scale_tensor[mask, 1] = temp

        w = rot_tensor[mask, 0]
        x = rot_tensor[mask, 1]
        y = rot_tensor[mask, 2]
        z = rot_tensor[mask, 3]

        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        w_new = (w - z) * inv_sqrt2
        x_new = (x + y) * inv_sqrt2
        y_new = (y - x) * inv_sqrt2
        z_new = (w + z) * inv_sqrt2

        rot_tensor[mask, 0] = w_new
        rot_tensor[mask, 1] = x_new
        rot_tensor[mask, 2] = y_new
        rot_tensor[mask, 3] = z_new

        norms = torch.norm(rot_tensor[mask], dim=-1, keepdim=True)
        rot_tensor[mask] = rot_tensor[mask] / norms

        # 同步修改 raw_vertex_data，保证 visualize 导出的 .ply 也一致。
        mask_np = mask.numpy()
        scale_0_name = scale_names[0]
        scale_1_name = scale_names[1]

        temp_s0 = raw_vertex_data[scale_0_name][mask_np].copy()
        raw_vertex_data[scale_0_name][mask_np] = raw_vertex_data[scale_1_name][mask_np]
        raw_vertex_data[scale_1_name][mask_np] = temp_s0

        rot_0_name = rot_names[0]
        rot_1_name = rot_names[1]
        rot_2_name = rot_names[2]
        rot_3_name = rot_names[3]

        raw_vertex_data[rot_0_name][mask_np] = rot_tensor[mask, 0].numpy()
        raw_vertex_data[rot_1_name][mask_np] = rot_tensor[mask, 1].numpy()
        raw_vertex_data[rot_2_name][mask_np] = rot_tensor[mask, 2].numpy()
        raw_vertex_data[rot_3_name][mask_np] = rot_tensor[mask, 3].numpy()

    id_to_idx = {int(id_val): i for i, id_val in enumerate(ids)}
    return gt_dict, id_to_idx, raw_vertex_data


def load_scene_meta(scene_dir):
    """
    读取相机 JSON 元信息，返回 seed_file -> meta 的映射。
    """
    meta_path = os.path.join(scene_dir, "train_cameras.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(scene_dir, "test_cameras.json")

    meta_dict = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
        for item in meta_data:
            meta_dict[item["seed_file"]] = item
    return meta_dict


def resolve_gt_ply_path(scene_dir, gt_ply_name):
    """
    解析当前 scene 使用哪一个 GT PLY。
    默认仍然读取 scene_dir/point_cloud.ply。
    如果传入绝对路径，则直接使用该路径。
    """
    if os.path.isabs(gt_ply_name):
        return gt_ply_name
    return os.path.join(scene_dir, gt_ply_name)


def build_frame_tensors(pt_path, filename, meta_dict, gt_dict, id_to_idx):
    """
    将单个 frame_xxx.pt 解析成训练输入 X 和监督 Y。
    返回：X_frame, Y_frame, valid_gt_indices
    """
    script_module = torch.jit.load(pt_path)
    data_list = list(script_module.parameters())

    if len(data_list) != 15:
        raise ValueError(f"Invalid number of parameters. Expected 15, got {len(data_list)}")

    (
        f_curr,
        render_f_curr,
        hist_f,
        hist_render_f,
        curr_dir,
        hist_dir,
        hist_mask,
        inv_depth,
        n_cam,
        mask_tensor,
        added_ids,
        R_wc,
        base_sh,
        base_pixel,
        dis,
    ) = data_list

    num_points_in_frame = added_ids.shape[0]

    f_curr = f_curr.view(num_points_in_frame, -1)
    render_f_curr = render_f_curr.view(num_points_in_frame, -1)
    hist_f = hist_f.view(num_points_in_frame, -1)
    hist_render_f = hist_render_f.view(num_points_in_frame, -1)
    curr_dir = curr_dir.view(num_points_in_frame, -1)
    hist_dir = hist_dir.view(num_points_in_frame, -1)
    hist_mask = hist_mask.view(num_points_in_frame, -1)

    inv_depth = inv_depth.view(num_points_in_frame, 1)
    n_cam = n_cam.view(num_points_in_frame, 3)
    mask_tensor = mask_tensor.view(num_points_in_frame, 1)
    R_wc = R_wc.view(num_points_in_frame, 9)
    base_sh = base_sh.view(num_points_in_frame, 3)
    base_pixel = base_pixel.view(num_points_in_frame, 2)
    dis = dis.view(num_points_in_frame, 1)

    if filename in meta_dict:
        item = meta_dict[filename]
        c_params = [item["fx"], item["fy"], item["cx"], item["cy"], float(item["width"]), float(item["height"])]
        c_pos = item["position"]
        cam_params = (
            torch.tensor(c_params, dtype=torch.float32, device=f_curr.device)
            .unsqueeze(0)
            .expand(num_points_in_frame, 6)
        )
        t_wc = (
            torch.tensor(c_pos, dtype=torch.float32, device=f_curr.device)
            .unsqueeze(0)
            .expand(num_points_in_frame, 3)
        )
    else:
        cam_params = torch.zeros((num_points_in_frame, 6), device=f_curr.device)
        t_wc = torch.zeros((num_points_in_frame, 3), device=f_curr.device)

    X_frame_full = torch.cat(
        [
            f_curr,
            render_f_curr,
            hist_f,
            hist_render_f,
            curr_dir,
            hist_dir,
            hist_mask,
            inv_depth,
            n_cam,
            mask_tensor,
            R_wc,
            base_sh,
            cam_params,
            t_wc,
            base_pixel,
            dis,
        ],
        dim=-1,
    )

    valid_indices = []
    gt_indices = []
    for idx in range(num_points_in_frame):
        id_val = int(added_ids[idx].item())
        if id_val in id_to_idx:
            valid_indices.append(idx)
            gt_indices.append(id_to_idx[id_val])

    if len(valid_indices) == 0:
        raise RuntimeError("All points were pruned or missing in GT")

    X_frame = X_frame_full[valid_indices]
    gt_indices_tensor = torch.tensor(gt_indices, dtype=torch.long)

    y_f_dc = gt_dict["f_dc"][gt_indices_tensor]
    y_f_rest = gt_dict["f_rest"][gt_indices_tensor]
    y_opacity = gt_dict["opacity"][gt_indices_tensor]
    y_scale = gt_dict["scale"][gt_indices_tensor]
    y_rot = gt_dict["rot"][gt_indices_tensor]
    y_xyz = gt_dict["xyz"][gt_indices_tensor]

    Y_frame = torch.cat([y_f_dc, y_f_rest, y_opacity, y_scale, y_rot, y_xyz], dim=-1)

    valid_point_mask = ~(
        torch.isnan(X_frame).any(dim=-1)
        | torch.isnan(Y_frame).any(dim=-1)
        | torch.isinf(X_frame).any(dim=-1)
        | torch.isinf(Y_frame).any(dim=-1)
    )

    if valid_point_mask.sum().item() == 0:
        raise RuntimeError("All points contain NaN or Inf")

    X_frame = X_frame[valid_point_mask]
    Y_frame = Y_frame[valid_point_mask]
    valid_gt_indices = gt_indices_tensor[valid_point_mask]
    return X_frame, Y_frame, valid_gt_indices


def maybe_visualize_frame(visualize, vis_dir, filename, raw_vertex_data, valid_gt_indices):
    """
    按需导出清洗后的 frame 级点云，便于调试数据质量。
    """
    if not visualize:
        return

    valid_gt_indices_np = valid_gt_indices.numpy()
    frame_raw_data = raw_vertex_data[valid_gt_indices_np]
    el = PlyElement.describe(frame_raw_data, "vertex")
    ply_filename = filename.replace(".pt", ".ply")
    ply_save_path = os.path.join(vis_dir, ply_filename)
    PlyData([el], text=False).write(ply_save_path)


def extract_scene_tensors(scene_dir, visualize=False, gt_ply_name="point_cloud.ply"):
    """
    原始路径：将整个 scene 的 X/Y 全部拼到内存。
    适合小数据，随机性最好，但大场景容易 OOM。
    """
    frames_dir = os.path.join(scene_dir, "frames")
    ply_path = resolve_gt_ply_path(scene_dir, gt_ply_name)

    if not os.path.isdir(frames_dir) or not os.path.exists(ply_path):
        print(f"Skipping {scene_dir}: Invalid dataset structure. Checking frameworks: {frames_dir}, {ply_path}")
        return None, None

    print(f"Loading GT from {ply_path}...")
    gt_dict, id_to_idx, raw_vertex_data = load_ply_gt(ply_path)
    meta_dict = load_scene_meta(scene_dir)

    vis_dir = None
    if visualize:
        vis_dir = os.path.join(scene_dir, "visualize")
        os.makedirs(vis_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".pt")])
    print(f"Processing {len(frame_files)} frames in {scene_dir}...")

    Xs = []
    Ys = []

    for filename in tqdm(frame_files):
        pt_path = os.path.join(frames_dir, filename)
        try:
            X_frame, Y_frame, valid_gt_indices = build_frame_tensors(
                pt_path,
                filename,
                meta_dict,
                gt_dict,
                id_to_idx,
            )
            maybe_visualize_frame(visualize, vis_dir, filename, raw_vertex_data, valid_gt_indices)
            Xs.append(X_frame)
            Ys.append(Y_frame)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

    if len(Xs) == 0:
        return None, None

    return torch.cat(Xs, dim=0), torch.cat(Ys, dim=0)


def collect_scene_to_binary(scene_dir, cache_dir, visualize=False, gt_ply_name="point_cloud.ply"):
    """
    将 scene 数据逐帧写入二进制文件，避免一次性堆在内存。

    返回：
    - x_bin_path, y_bin_path: 二进制文件路径
    - num_rows: 总样本数
    - x_dim, y_dim: 特征维度
    """
    frames_dir = os.path.join(scene_dir, "frames")
    ply_path = resolve_gt_ply_path(scene_dir, gt_ply_name)

    if not os.path.isdir(frames_dir) or not os.path.exists(ply_path):
        print(f"Skipping {scene_dir}: Invalid dataset structure. Checking frameworks: {frames_dir}, {ply_path}")
        return None, None, 0, 0, 0

    print(f"Loading GT from {ply_path}...")
    gt_dict, id_to_idx, raw_vertex_data = load_ply_gt(ply_path)
    meta_dict = load_scene_meta(scene_dir)

    vis_dir = None
    if visualize:
        vis_dir = os.path.join(scene_dir, "visualize")
        os.makedirs(vis_dir, exist_ok=True)

    os.makedirs(cache_dir, exist_ok=True)
    x_bin_path = os.path.join(cache_dir, "X_scene.bin")
    y_bin_path = os.path.join(cache_dir, "Y_scene.bin")

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".pt")])
    print(f"Processing {len(frame_files)} frames in {scene_dir}...")

    num_rows = 0
    x_dim = 0
    y_dim = 0

    with open(x_bin_path, "wb") as fx, open(y_bin_path, "wb") as fy:
        for filename in tqdm(frame_files):
            pt_path = os.path.join(frames_dir, filename)
            try:
                X_frame, Y_frame, valid_gt_indices = build_frame_tensors(
                    pt_path,
                    filename,
                    meta_dict,
                    gt_dict,
                    id_to_idx,
                )
                maybe_visualize_frame(visualize, vis_dir, filename, raw_vertex_data, valid_gt_indices)

                X_np = X_frame.detach().cpu().numpy().astype(np.float32, copy=False)
                Y_np = Y_frame.detach().cpu().numpy().astype(np.float32, copy=False)

                if x_dim == 0:
                    x_dim = X_np.shape[1]
                    y_dim = Y_np.shape[1]

                # 逐帧顺序写入磁盘，避免把整场景堆在内存。
                X_np.tofile(fx)
                Y_np.tofile(fy)
                num_rows += X_np.shape[0]
            except Exception as e:
                print(f"Skipping {filename}: {e}")
                continue

    return x_bin_path, y_bin_path, num_rows, x_dim, y_dim


def save_chunks_from_tensor(X_data, Y_data, prefix, bucket_size, target_dir, start_bucket_idx, scene_name):
    """
    从内存张量按 bucket_size 切分保存。
    """
    num_points = X_data.shape[0]
    if num_points == 0:
        return start_bucket_idx, 0

    num_buckets = (num_points + bucket_size - 1) // bucket_size
    points_saved = 0

    for i in range(num_buckets):
        start = i * bucket_size
        end = min((i + 1) * bucket_size, num_points)

        X_chunk = X_data[start:end].clone()
        Y_chunk = Y_data[start:end].clone()

        bucket_name = f"{prefix}_bucket_{start_bucket_idx:04d}_{scene_name}.pt"
        bucket_path = os.path.join(target_dir, bucket_name)
        torch.save({"X": X_chunk, "Y": Y_chunk}, bucket_path)

        start_bucket_idx += 1
        points_saved += (end - start)

    return start_bucket_idx, points_saved


def save_chunks_from_memmap(
    x_memmap,
    y_memmap,
    indices,
    prefix,
    bucket_size,
    target_dir,
    start_bucket_idx,
    scene_name,
):
    """
    从 memmap 中按“随机索引”切片保存桶。
    这里的随机性来自 indices 的顺序。
    """
    num_points = len(indices)
    if num_points == 0:
        return start_bucket_idx, 0

    num_buckets = (num_points + bucket_size - 1) // bucket_size
    points_saved = 0

    for i in range(num_buckets):
        start = i * bucket_size
        end = min((i + 1) * bucket_size, num_points)
        batch_indices = indices[start:end]

        # 为了减少 memmap 的随机读开销：
        # 1) 先对索引排序，尽量顺序读取；
        # 2) 读取完成后再在桶内随机打乱，保证每个桶内部依然是随机分布。
        sorted_indices = np.sort(batch_indices)
        X_chunk_np = np.asarray(x_memmap[sorted_indices], dtype=np.float32)
        Y_chunk_np = np.asarray(y_memmap[sorted_indices], dtype=np.float32)

        if X_chunk_np.shape[0] > 1:
            local_perm = np.random.permutation(X_chunk_np.shape[0])
            X_chunk_np = X_chunk_np[local_perm]
            Y_chunk_np = Y_chunk_np[local_perm]

        X_chunk = torch.from_numpy(X_chunk_np)
        Y_chunk = torch.from_numpy(Y_chunk_np)

        bucket_name = f"{prefix}_bucket_{start_bucket_idx:04d}_{scene_name}.pt"
        bucket_path = os.path.join(target_dir, bucket_name)
        torch.save({"X": X_chunk, "Y": Y_chunk}, bucket_path)

        if (i + 1) % 1 == 0:
            print(
                f"  [{prefix}] bucket {i + 1}/{num_buckets} saved "
                f"({end}/{num_points} points)"
            )

        start_bucket_idx += 1
        points_saved += (end - start)

    return start_bucket_idx, points_saved


def process_scene_in_memory(
    scene_dir,
    scene_name,
    shards_dir,
    val_shards_dir,
    bucket_size,
    val_ratio,
    visualize,
    train_bucket_idx,
    val_bucket_idx,
    gt_ply_name,
):
    """
    原始逻辑：场景数据全进内存后再打乱。
    """
    X_scene, Y_scene = extract_scene_tensors(scene_dir, visualize, gt_ply_name)
    if X_scene is None or X_scene.shape[0] == 0:
        return train_bucket_idx, val_bucket_idx, 0, 0

    num_points = X_scene.shape[0]
    print(f"Successfully extracted {num_points} points from {scene_name}.")
    print("Performing Scene-Level Shuffle...")

    perm = torch.randperm(num_points)
    X_scene = X_scene[perm]
    Y_scene = Y_scene[perm]

    val_size = int(num_points * val_ratio)
    train_size = num_points - val_size

    X_val = X_scene[:val_size]
    Y_val = Y_scene[:val_size]
    X_train = X_scene[val_size:]
    Y_train = Y_scene[val_size:]

    print(f" -> Splitting: {train_size} Train points, {val_size} Val points (ratio={val_ratio})")

    print(f"Chunking Train buckets of size {bucket_size}...")
    train_bucket_idx, train_points = save_chunks_from_tensor(
        X_train,
        Y_train,
        "train",
        bucket_size,
        shards_dir,
        train_bucket_idx,
        scene_name,
    )

    val_points = 0
    if val_size > 0:
        print(f"Chunking Val buckets of size {bucket_size}...")
        val_bucket_idx, val_points = save_chunks_from_tensor(
            X_val,
            Y_val,
            "val",
            bucket_size,
            val_shards_dir,
            val_bucket_idx,
            scene_name,
        )

    del X_scene, Y_scene, X_train, Y_train, X_val, Y_val
    return train_bucket_idx, val_bucket_idx, train_points, val_points


def process_scene_external_shuffle(
    scene_dir,
    scene_name,
    shards_dir,
    val_shards_dir,
    bucket_size,
    val_ratio,
    visualize,
    cache_root,
    train_bucket_idx,
    val_bucket_idx,
    gt_ply_name,
):
    """
    外部打乱（低内存）逻辑：
    1) 逐帧写入磁盘二进制
    2) 用 memmap 映射整个场景
    3) 生成全局随机索引并分桶保存
    """
    scene_cache_dir = os.path.join(cache_root, scene_name)
    os.makedirs(scene_cache_dir, exist_ok=True)

    x_bin_path, y_bin_path, num_points, x_dim, y_dim = collect_scene_to_binary(
        scene_dir=scene_dir,
        cache_dir=scene_cache_dir,
        visualize=visualize,
        gt_ply_name=gt_ply_name,
    )

    if num_points == 0:
        print(f"No valid points in {scene_name}, skip.")
        return train_bucket_idx, val_bucket_idx, 0, 0

    print(f"Successfully extracted {num_points} points from {scene_name} (external shuffle mode).")
    print("Performing Scene-Level Shuffle with global random indices...")

    x_memmap = np.memmap(x_bin_path, dtype=np.float32, mode="r", shape=(num_points, x_dim))
    y_memmap = np.memmap(y_bin_path, dtype=np.float32, mode="r", shape=(num_points, y_dim))

    # 这里仍然是“全场景级随机”。只是不再把全量特征堆在内存。
    perm = np.random.permutation(num_points)

    val_size = int(num_points * val_ratio)
    train_size = num_points - val_size
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    print(f" -> Splitting: {train_size} Train points, {val_size} Val points (ratio={val_ratio})")

    print(f"Chunking Train buckets of size {bucket_size}...")
    train_bucket_idx, train_points = save_chunks_from_memmap(
        x_memmap,
        y_memmap,
        train_indices,
        "train",
        bucket_size,
        shards_dir,
        train_bucket_idx,
        scene_name,
    )

    val_points = 0
    if val_size > 0:
        print(f"Chunking Val buckets of size {bucket_size}...")
        val_bucket_idx, val_points = save_chunks_from_memmap(
            x_memmap,
            y_memmap,
            val_indices,
            "val",
            bucket_size,
            val_shards_dir,
            val_bucket_idx,
            scene_name,
        )

    # 主动释放 memmap 句柄，并清理临时缓存。
    del x_memmap, y_memmap
    try:
        os.remove(x_bin_path)
        os.remove(y_bin_path)
        os.rmdir(scene_cache_dir)
    except OSError:
        pass

    return train_bucket_idx, val_bucket_idx, train_points, val_points


def main(
    root_dir,
    shards_dir,
    val_shards_dir,
    bucket_size,
    val_ratio=0.10,
    visualize=False,
    external_shuffle=False,
    cache_dir=None,
    gt_ply_name="point_cloud.ply",
):
    print(f"Starting Sharding preprocessing from root: {root_dir}")
    os.makedirs(shards_dir, exist_ok=True)
    if val_ratio > 0:
        os.makedirs(val_shards_dir, exist_ok=True)

    if external_shuffle:
        if cache_dir is None:
            cache_dir = os.path.join(root_dir, "_scene_cache")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"External shuffle cache dir: {cache_dir}")

    train_bucket_idx = 0
    val_bucket_idx = 0
    total_train_points = 0
    total_val_points = 0

    scene_items = sorted(os.listdir(root_dir))
    for item in scene_items:
        scene_dir = os.path.join(root_dir, item)
        if not (os.path.isdir(scene_dir) and item != "shards" and "dataset-" in item):
            continue

        print("\n=====================================")
        print(f" processing scene: {item}")
        print("=====================================")

        if external_shuffle:
            train_bucket_idx, val_bucket_idx, train_pts, val_pts = process_scene_external_shuffle(
                scene_dir=scene_dir,
                scene_name=item,
                shards_dir=shards_dir,
                val_shards_dir=val_shards_dir,
                bucket_size=bucket_size,
                val_ratio=val_ratio,
                visualize=visualize,
                cache_root=cache_dir,
                train_bucket_idx=train_bucket_idx,
                val_bucket_idx=val_bucket_idx,
                gt_ply_name=gt_ply_name,
            )
        else:
            train_bucket_idx, val_bucket_idx, train_pts, val_pts = process_scene_in_memory(
                scene_dir=scene_dir,
                scene_name=item,
                shards_dir=shards_dir,
                val_shards_dir=val_shards_dir,
                bucket_size=bucket_size,
                val_ratio=val_ratio,
                visualize=visualize,
                train_bucket_idx=train_bucket_idx,
                val_bucket_idx=val_bucket_idx,
                gt_ply_name=gt_ply_name,
            )

        total_train_points += train_pts
        total_val_points += val_pts

    print("\nPreprocessing Complete!")
    print(f"Total Train Points: {total_train_points} in {train_bucket_idx} buckets (saved in {shards_dir})")
    print(f"Total Val Points  : {total_val_points} in {val_bucket_idx} buckets (saved in {val_shards_dir})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create randomly sharded uniform bucket files for streaming dataloader."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/lab404/dataset/gs_rg_r3live",
        help="Path containing dataset-* scenes",
    )
    parser.add_argument(
        "--shards_dir",
        type=str,
        default="/home/lab404/dataset/gs_rg_r3live/shards",
        help="Directory to save train bucket_*.pt files",
    )
    parser.add_argument(
        "--val_shards_dir",
        type=str,
        default="/home/lab404/dataset/gs_rg_r3live/val_shards",
        help="Directory to save validation bucket_*.pt files",
    )
    parser.add_argument(
        "--bucket_size",
        type=int,
        default=250000,
        help="Number of Gaussian points per bucket file",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
        help="Validation split ratio (0.0~1.0)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save individual cleaned frame point clouds",
    )
    parser.add_argument(
        "--external_shuffle",
        action="store_true",
        help="Use disk-based external shuffle to reduce peak memory",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory used only when --external_shuffle is enabled",
    )
    parser.add_argument(
        "--gt_ply_name",
        type=str,
        default="point_cloud.ply",
        help="GT PLY file name under each scene, or an absolute path",
    )

    args = parser.parse_args()
    main(
        root_dir=args.root_dir,
        shards_dir=args.shards_dir,
        val_shards_dir=args.val_shards_dir,
        bucket_size=args.bucket_size,
        val_ratio=args.val_ratio,
        visualize=args.visualize,
        external_shuffle=args.external_shuffle,
        cache_dir=args.cache_dir,
        gt_ply_name=args.gt_ply_name,
    )
