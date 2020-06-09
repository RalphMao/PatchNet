import numpy as np
import torch 

def find_peaks(score_maps):
    assert len(score_maps.shape) == 3
    b, h, w = score_maps.shape
    original_device = score_maps.device
    # Preprocess
    # score_maps = torch.relu(score_maps - score_maps.view(b, -1).max(dim=1)[0][:, None, None] + 2)
    score_maps = torch.relu(score_maps - score_maps.view(b, -1).max(dim=1)[0][:, None, None] * 0.75)
    score_maps = score_maps.cpu().numpy()
    scores = torch.zeros(b)
    locs = torch.zeros(b).long()
    for idx in range(b):
        peak_locs, peak_scores = find_peaks_2d(score_maps[idx])
        normalized_score = peak_scores.max() / (1e-9 + peak_scores.sum())
        peak_loc = peak_locs[np.argmax(peak_scores)]
        locs[idx] = int(peak_loc)
        scores[idx] = normalized_score
    return scores.to(original_device), locs.to(original_device)

def find_peaks_2d(score_map):
    H, W = score_map.shape
    peak_flags = np.zeros_like(score_map)
    for h in range(score_map.shape[0]):
        for w in range(score_map.shape[1]):
            neighbors = score_map[max(0, h-1):h+2, max(0, w-1):w+2]
            peak_flags[h, w] = np.sum(score_map[h, w] <= neighbors) == 1

    if peak_flags.sum() == 0:
        peak_scores = np.zeros(1)
        peak_locs = np.ones(1, dtype='i') * np.argmax(score_map)
    else:
        peak_locs = np.where(peak_flags.flatten())[0]
        peak_scores = np.zeros(len(peak_locs))
        for idx, loc in enumerate(peak_locs):
            h = loc // W
            w = loc % W
            peak_scores[idx] = calculate_peak_score(score_map, h, w)
    return peak_locs, peak_scores
    
def calculate_peak_score(score_map, h, w):
    return score_map[h, w]

def cuttop(input, top):
    return top - torch.relu(top - input)

def conv2gconv(conv_weights, groups):
    cout, cin, k, _ = conv_weights.shape
    assert cout % groups == 0 and cin % groups == 0
    gcin = cin // groups
    gcout = cout // groups
    gconv_weights = torch.zeros(cout, gcin, k, k).to(conv_weights.device)
    for g in range(groups):
        gconv_weights[gcout * g:gcout * (g+1)] = conv_weights[gcout * g:gcout * (g+1), gcin * g: gcin * (g+1)]
    return gconv_weights

def tile2d(data, dim):
    size = int(np.sqrt(data.shape[dim]))
    logsize = int(np.log2(size))
    assert size * size == data.shape[dim]
    assert np.exp2(logsize) == size
    extended_dim = dim + logsize * 2 - 1
    extended_ndim = len(data.shape) + logsize * 2 - 1
    data_shape = tuple(data.shape)
    new_shape = data_shape[:dim] + (2,) * 2 * logsize + data_shape[dim+1:]
    permute_order = list(range(dim)) + [dim + (idx % 2) * logsize + idx // 2 for idx in range(2 * logsize)] + list(range(extended_dim+1, extended_ndim))

    tiled_data = data.view(new_shape).permute(*permute_order).contiguous().view(data_shape)
    return tiled_data


