import torch
import torch.nn as nn
import numpy as np
import math

def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t

    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)

    return pos_x

def nerf_positional_encoding(tensor, num_encoding_functions=6, include_input=False, log_sampling=True) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)    

def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape
        
        view_shape = [1 for _ in range(len(feat_shape))] 
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)
        
        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat

def get_dis_point2point(point, point_=(0.0, 0.0)):
    return np.sqrt(np.square((point[0] - point_[0])) + np.square((point[1] - point_[1])))

def get_aux_lines_b2d(gt_vecs_pts, vectorizemap_dist, device):
    aux_lines = []
    thres = 0.1
    for gt_lines_b1 in gt_vecs_pts:
        aux_lines_b1 = []
        if len(gt_lines_b1) == 0:
            aux_lines.append(aux_lines_b1)
            continue
        gt_lines_b1 = gt_lines_b1.squeeze(1).cpu().detach()
        gt_lines_b1_numpy = [instance.numpy() for instance in gt_lines_b1]
        # filter
        gt_lines_b1_numpy_f0 = []
        for ii in range(len(gt_lines_b1_numpy)):
            if get_dis_point2point(gt_lines_b1_numpy[ii][0], gt_lines_b1_numpy[ii][-1]) < thres:
                continue
            gt_lines_b1_numpy_f0.append(gt_lines_b1_numpy[ii])
        gt_lines_b1_numpy = gt_lines_b1_numpy_f0

        gt_lines_b1_numpy_f1 = []
        for ii in range(len(gt_lines_b1_numpy)):
            overlap = False
            for jj in range(len(gt_lines_b1_numpy_f1)):
                if get_dis_point2point(gt_lines_b1_numpy[ii][0], gt_lines_b1_numpy_f1[jj][0]) < thres and \
                   get_dis_point2point(gt_lines_b1_numpy[ii][-1], gt_lines_b1_numpy_f1[jj][-1]) < thres:
                    overlap = True
                    break
            if not overlap:
                gt_lines_b1_numpy_f1.append(gt_lines_b1_numpy[ii])
        gt_lines_b1_numpy = gt_lines_b1_numpy_f1

        # merge lines #
        start_line_ids = []
        for ii in range(len(gt_lines_b1_numpy)):
            start_line = True
            for jj in range(len(gt_lines_b1_numpy)):
                # begin pt is not a end pt of another line
                if get_dis_point2point(gt_lines_b1_numpy[ii][0], gt_lines_b1_numpy[jj][-1]) < thres:
                    start_line = False
                    break
            if start_line:
                start_line_ids.append(ii)
        
        def find_next(current_merged_line, used_index):
            new_merged_lines = []
            new_used_indexs = []

            next_lines = []
            next_index = []
            for jj in range(len(gt_lines_b1_numpy)):
                if get_dis_point2point(current_merged_line[-1], gt_lines_b1_numpy[jj][0]) < thres:
                    if jj not in used_index:
                        next_lines.append(gt_lines_b1_numpy[jj])
                        next_index.append(jj)
            
            if len(next_lines) == 0:
                return [current_merged_line], [used_index]
            for next_line, line_index in zip(next_lines, next_index):
                new_merged_line = np.concatenate([current_merged_line, next_line])
                new_used_index = used_index + [line_index]
                new_merged_line, new_used_index = find_next(new_merged_line, new_used_index)
                new_merged_lines += new_merged_line
                new_used_indexs += new_used_index
                
            return new_merged_lines, new_used_indexs
        
        merged_lines = []
        for ii in start_line_ids:
            current_merged_lines, _ = find_next(gt_lines_b1_numpy[ii], [ii])
            merged_lines += current_merged_lines
        
        # dist lines #
        dist_lines = vectorizemap_dist(merged_lines)
        for line_dix, line in enumerate(dist_lines):
            if len(line) < 2:
                line = merged_lines[line_dix]
            line = torch.tensor(line, dtype=torch.float32, device=device)
            aux_lines_b1.append(line)
        aux_lines.append(aux_lines_b1)

    return aux_lines

def get_aux_lines_v1(gt_vecs_pts, vectorizemap_dist, device):
    aux_lines = []
    for gt_lines_b1 in gt_vecs_pts:
        aux_lines_b1 = []
        if len(gt_lines_b1.instance_list) == 0:
            aux_lines.append(aux_lines_b1)
            continue
        gt_lines_b1_numpy = [np.array(list(instance.coords)) for instance in gt_lines_b1.instance_list]
        # merge lines #
        start_line_ids = []
        for ii in range(len(gt_lines_b1_numpy)):
            start_line = True
            for jj in range(len(gt_lines_b1_numpy)):
                # begin pt is not a end pt of another line
                if get_dis_point2point(gt_lines_b1_numpy[ii][0], gt_lines_b1_numpy[jj][-1]) < 0.1:
                    start_line = False
                    break
            if start_line:
                start_line_ids.append(ii)
        
        def find_next(current_merged_line, used_index):
            new_merged_lines = []
            new_used_indexs = []

            next_lines = []
            next_index = []
            for jj in range(len(gt_lines_b1_numpy)):
                if get_dis_point2point(current_merged_line[-1], gt_lines_b1_numpy[jj][0]) < 0.1:
                    if jj not in used_index:
                        next_lines.append(gt_lines_b1_numpy[jj])
                        next_index.append(jj)
            
            if len(next_lines) == 0:
                return [current_merged_line], [used_index]
            for next_line, line_index in zip(next_lines, next_index):
                new_merged_line = np.concatenate([current_merged_line, next_line])
                new_used_index = used_index + [line_index]
                new_merged_line, new_used_index = find_next(new_merged_line, new_used_index)
                new_merged_lines += new_merged_line
                new_used_indexs += new_used_index
                
            return new_merged_lines, new_used_indexs
        
        merged_lines = []
        for ii in start_line_ids:
            current_merged_lines, _ = find_next(gt_lines_b1_numpy[ii], [ii])
            merged_lines += current_merged_lines
        
        # dist lines #
        dist_lines = vectorizemap_dist(merged_lines)
        for line_dix, line in enumerate(dist_lines):
            if len(line) < 2:
                line = merged_lines[line_dix]
            line = torch.tensor(line, dtype=torch.float32, device=device)
            aux_lines_b1.append(line)
        aux_lines.append(aux_lines_b1)

    return aux_lines

def get_aux_lines_v2(gt_vecs_pts, vectorizemap_dist, device):
    aux_lines = []
    for gt_lines_b1 in gt_vecs_pts:
        aux_lines_b1 = []
        if len(gt_lines_b1.instance_list) == 0:
            aux_lines.append(aux_lines_b1)
            continue
        gt_lines_b1_numpy = [np.array(list(instance.coords)) for instance in gt_lines_b1.instance_list]
        adj_matrix = gt_lines_b1.adj_matrix
        adj_matrix = np.where(adj_matrix)
        adj_matrix = np.stack(adj_matrix).T
        # merge lines #
        start_line_ids = []
        for ii in range(len(gt_lines_b1_numpy)):
            start_line = True
            for jj in range(len(adj_matrix)):
                # begin pt is not a end pt of another line
                if adj_matrix[jj][1] == ii:
                    start_line = False
                    break
            if start_line:
                start_line_ids.append(ii)
        
        def find_next(current_merged_line, used_index):
            new_merged_lines = []
            new_used_indexs = []

            next_lines = []
            next_index = []
            for jj in range(len(adj_matrix)):
                if used_index[-1] == adj_matrix[jj][0]:
                    nidx = adj_matrix[jj][1]
                    if nidx not in used_index:
                        next_lines.append(gt_lines_b1_numpy[nidx])
                        next_index.append(nidx)
            
            if len(next_lines) == 0:
                return [current_merged_line], [used_index]
            for next_line, line_index in zip(next_lines, next_index):
                new_merged_line = np.concatenate([current_merged_line, next_line])
                new_used_index = used_index + [line_index]
                new_merged_line, new_used_index = find_next(new_merged_line, new_used_index)
                new_merged_lines += new_merged_line
                new_used_indexs += new_used_index
                
            return new_merged_lines, new_used_indexs
        
        merged_lines = []
        for ii in start_line_ids:
            current_merged_lines, _ = find_next(gt_lines_b1_numpy[ii], [ii])
            merged_lines += current_merged_lines
        
        # dist lines #
        dist_lines = vectorizemap_dist(merged_lines)
        for line_dix, line in enumerate(dist_lines):
            if len(line) < 2:
                line = merged_lines[line_dix]
            line = torch.tensor(line, dtype=torch.float32, device=device)
            aux_lines_b1.append(line)
        aux_lines.append(aux_lines_b1)

    return aux_lines