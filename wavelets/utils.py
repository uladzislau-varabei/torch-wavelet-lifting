import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


NCHW_FORMAT = 'NCHW'
NHWC_FORMAT = 'NHWC'
to_NHWC_axis = [0, 2, 3, 1] # NCHW -> NHWC
to_NCHW_axis = [0, 3, 1, 2] # NHWC -> NCHW

# DEFAULT_DATA_FORMAT = NCHW_FORMAT
DEFAULT_DATA_FORMAT = NHWC_FORMAT # should now work faster

PAD_MODE = 'constant'

# 2d transform, so power 2
COEFFS_SCALES_2D_v1 = np.array([
    1 / np.sqrt(2),
    np.sqrt(2),
    np.sqrt(2),
    np.sqrt(2)
], dtype=np.float32) ** 2

# The same scales allows to get coeffs ranges that are consistent
COEFFS_SCALES_2D_v2 = np.array([
    1 / np.sqrt(2) ** 2,
    1 / np.sqrt(2) ** 2,
    1 / np.sqrt(2) ** 2,
    1 / np.sqrt(2) ** 2
], dtype=np.float32)

# 2d transform, so use double power only for LL coeffs
COEFFS_SCALES_2D_v3 = np.array([
    1 / np.sqrt(2) ** 2,
    1 / np.sqrt(2),
    1 / np.sqrt(2),
    1 / np.sqrt(2)
], dtype=np.float32)

COEFFS_SCALES_2D_v4 = np.array([
    1 / np.sqrt(2),
    1,
    1,
    1
], dtype=np.float32)

COEFFS_SCALES_2D_v5 = np.array([
    1 / np.sqrt(2),
    1,
    1,
    np.sqrt(2)
], dtype=np.float32)

# LL taken from v3, H coeffs from v5
COEFFS_SCALES_2D_v6 = np.array([
    1 / np.sqrt(2) ** 2,
    1,
    1,
    np.sqrt(2)
], dtype=np.float32)

COEFFS_SCALES_2D_DICT = {
    1: COEFFS_SCALES_2D_v1,
    2: COEFFS_SCALES_2D_v2,
    3: COEFFS_SCALES_2D_v3,
    4: COEFFS_SCALES_2D_v4,
    5: COEFFS_SCALES_2D_v5,
    6: COEFFS_SCALES_2D_v6
}

# 6 is the best for preserving source data range for LL and keeping similar ranges for all H details
# Found with tests.py with and without normalization fro input
COEFFS_SCALES_V = 6
COEFFS_SCALES_2D = torch.from_numpy(COEFFS_SCALES_2D_DICT[COEFFS_SCALES_V])

DEFAULT_SCALE_1D_COEFFS = True
DEFAULT_SCALE_2d_COEFFS = True


# ----- Utils -----

def scale_into_range(x, target_range):
    src_range = (x.min(), x.max())
    src_range_size = src_range[1] - src_range[0]
    target_range_size = target_range[1] - target_range[0]
    x = (x - src_range[0]) * target_range_size / src_range_size + target_range[0]
    return x


def eval_stats_dict(x, name):
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    round_digits = 3
    return {
        f'{name}_min': round(x_np.min(), round_digits),
        f'{name}_max': round(x_np.max(), round_digits),
        f'{name}_mean': round(x_np.mean(), round_digits),
        f'{name}_abs_mean': round(np.abs(x_np).mean(), round_digits),
        f'{name}_unit_energy': round(np.sqrt((x_np ** 2).sum()).mean(), round_digits)
    }


def eval_stats(x):
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    round_digits = 3
    x_min = round(x_np.min(), round_digits)
    x_max = round(x_np.max(), round_digits)
    x_mean = round(x_np.mean(), round_digits)
    x_abs_mean = round(np.abs(x_np).mean(), round_digits)
    q_delta = 10
    x_q1 = round(np.percentile(x_np, q=q_delta), round_digits)
    x_q2 = round(np.percentile(x_np, q=100 - q_delta), round_digits)
    return x_min, x_max, x_mean, x_abs_mean, x_q1, x_q2

def test_lifting_scheme(image, kernel, forward_2d_op, backward_2d_op, scale_1d_coefs=True, scale_2d_coefs=True,
                        data_format=DEFAULT_DATA_FORMAT, print_logs=True):
    if data_format == NCHW_FORMAT:
        image = np.transpose(image, (2, 0, 1))

    input_image = image[None, ...].astype(np.float32)
    input_image = (input_image / 127.5) - 1.0
    if print_logs:
        print(f'Input image min: {input_image.min()}, max: {input_image.max()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_image = torch.from_numpy(input_image).to(device)
    coeffs_scales_2d = COEFFS_SCALES_2D

    anz_image = forward_2d_op(input_image, kernel,
                              scale_1d_coeffs=scale_1d_coefs,
                              scale_2d_coeffs=scale_2d_coefs,
                              coeffs_scales_2d=coeffs_scales_2d,
                              data_format=data_format)
    if print_logs:
        print(f'Input image shape: {input_image.shape}, anz image shape: {anz_image.shape}')
    # Apply deepcopy as in-placed ops are used and the same tensor is used later
    restored_image = backward_2d_op(copy.deepcopy(anz_image), kernel,
                                    scale_1d_coeffs=scale_1d_coefs,
                                    scale_2d_coeffs=scale_2d_coefs,
                                    coeffs_scales_2d=coeffs_scales_2d,
                                    data_format=data_format)
    diffs = (input_image - restored_image).detach().cpu().numpy()
    error = (diffs.flatten() ** 2).mean()

    anz_image_coeffs = extract_coeffs_from_channels(anz_image, data_format=data_format)
    scaled_anz_image_coeffs = []
    scales = eval_stats_dict(input_image, 'src')  # add to this dict coeffs stats later
    coeffs_names = ['x_LL', 'x_LH', 'X_HL', 'X_HH']
    for idx, c in enumerate(anz_image_coeffs):
        if scale_2d_coefs:
            scaled_c = c
        else:
            scaled_c = coeffs_scales_2d[idx] * c
        name = coeffs_names[idx]
        vis_c = scale_into_range(c, (0, 1))
        if print_logs:
            print(f'{name}: src min = {c.min():.3f}, max = {c.max():.3f}, scaled min = {scaled_c.min():.3f}, scaled max = {scaled_c.max():.3f}')
        scaled_anz_image_coeffs.append(vis_c)
        coeffs_scales = eval_stats_dict(c, name)
        scales = {**scales, **coeffs_scales}
    vis_anz_image = merge_coeffs_into_spatial(scaled_anz_image_coeffs, data_format=data_format)
    vis_anz_image = vis_anz_image[0].detach().cpu().numpy()
    if data_format == NCHW_FORMAT:
        vis_anz_image = np.transpose(vis_anz_image, (1, 2, 0))
    vis_anz_image = (255 * vis_anz_image).astype(np.uint8)
    if print_logs:
        print(f'Analysis/synthesis error: {error}')
    return vis_anz_image, error, scales

def test_lifting_scales(image, name, kernel, forward_2d_op, normalize_input=True, data_format=DEFAULT_DATA_FORMAT,
                        plot_data=True, plot_hist=True):
    import matplotlib.pyplot as plt

    if data_format == NCHW_FORMAT:
        image = np.transpose(image, (2, 0, 1))

    input_image = image[None, ...].astype(np.float32)
    if normalize_input:
        input_image = (input_image / 127.5) - 1.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_image = torch.from_numpy(input_image).to(device)

    stats = {}
    for scales_v in sorted(list(COEFFS_SCALES_2D_DICT.keys())):
        coeffs_scales_2d = torch.from_numpy(COEFFS_SCALES_2D_DICT[scales_v])
        anz_image = forward_2d_op(input_image, kernel,
                                  scale_1d_coeffs=True,
                                  scale_2d_coeffs=True,
                                  coeffs_scales_2d=coeffs_scales_2d,
                                  data_format=data_format)
        anz_image_coeffs = extract_coeffs_from_channels(anz_image, data_format=data_format)
        coeffs_names = ['x_LL', 'x_LH', 'X_HL', 'X_HH']
        if plot_data:
            fig, ax = plt.subplots(nrows=5, ncols=1)
            fig.suptitle(f'Scales_2d v={scales_v}, {name}')
        data = input_image.detach().cpu().numpy().flatten()
        label = 'Src image'
        if plot_data:
            if plot_hist:
                q_delta = 15
                range_min = np.percentile(data, q=q_delta)
                range_max = np.percentile(data, q=100 - q_delta)
                ax[0].hist(data, bins=100, range=(range_min, range_max), density=True, label=label)
            else:
                ax[0].plot(data, label=label)
            ax[0].legend()
        scale_stats = []
        scale_stats.append(eval_stats(data))
        for idx, c in enumerate(anz_image_coeffs, 1):
            data = c.detach().cpu().numpy().flatten()
            scale_stats.append(eval_stats(data))
            label = coeffs_names[idx - 1]
            if plot_data:
                if plot_hist:
                    ax[idx].hist(data, bins=100, density=True, label=label)
                else:
                    ax[idx].plot(data, label=label)
                ax[idx].legend()
        df_columns = ['min', 'max', 'mean', 'abs_mean', 'q1', 'q2']
        stats[scales_v] = pd.DataFrame(scale_stats, columns=df_columns, index=['src'] + coeffs_names)

    print('Stats:')
    for k in sorted(list(stats.keys())):
        df = stats[k]
        print(f'v={k}:\n{df}')
    if plot_data:
        plt.show()


# ----- Merging/splitting coeffs -----

def prepare_coeffs_for_1d_op(x, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
    assert not (across_cols and across_rows) and (across_cols or across_rows)
    # o - odd, e - even
    if data_format == NCHW_FORMAT:
        if across_cols:
            # Inputs have shape NCHW and operation is applied across W axis (cols)
            x_ev_0 = x[:, :, :, 0::2]
            x_od_0 = x[:, :, :, 1::2]
        else: # across_rows:
            # Inputs have shape NCHW and operation is applied across H axis (rows)
            x_ev_0 = x[:, :, 0::2, :]
            x_od_0 = x[:, :, 1::2, :]
    else:  # data_format == NHWC_FORMAT:
        if across_cols:
            # Inputs have shape NHWC and operation is applied across W axis (cols)
            x_ev_0 = x[:, :, 0::2, :]
            x_od_0 = x[:, :, 1::2, :]
        else: # across_rows:
            # Inputs have shape NHWC and operation is applied across H axis (rows)
            x_ev_0 = x[:, 0::2, :, :]
            x_od_0 = x[:, 1::2, :, :]
    return (x_ev_0, x_od_0)


def prepare_coeffs_for_inv_1d_op(x_coefs, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
    assert not (across_cols and across_rows) and (across_cols or across_rows)
    # x_coefs: s, d
    if data_format == NCHW_FORMAT:
        _, C, H, W = x_coefs.shape
        if across_cols:
            # Inputs have shape NCHW and operation is applied across W axis (cols)
            s, d = x_coefs[:, :, :, : W // 2], x_coefs[:, :, :, W // 2:]
        else:  # across_rows:
            # Inputs have shape NCHW and operation is applied across H axis (rows)
            s, d = x_coefs[:, :, : H // 2, :], x_coefs[:, :, H // 2:, :]
    else:  # if data_format == NHWC_FORMAT:
        _, H, W, C = x_coefs.shape
        if across_cols:
            # Inputs have shape NHWC and operation is applied across W axis (cols)
            s, d = x_coefs[:, :, : W // 2, :], x_coefs[:, :, W // 2:, :]
        else:  # across_rows:
            # Inputs have shape NHWC and operation is applied across H axis (rows)
            s, d = x_coefs[:, : H // 2, :, :], x_coefs[:, H // 2:, :, :]
    return (s, d)


def join_coeffs_after_1d_op(coeffs, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
    assert not (across_cols and across_rows) and (across_cols or across_rows)
    x_s, x_d = coeffs
    if across_cols:
        if data_format == NCHW_FORMAT:
            # Shapes of x_s and x_d here: [N, C, H, W // 2]
            x = torch.cat([x_s, x_d], dim=3)
        else:  # data_format == NHWC_FORMAT:
            # Shapes of x_s and x_d here: [N, H, W // 2, C]
            x = torch.cat([x_s, x_d], dim=2)
    else: # across_rows:
        if data_format == NCHW_FORMAT:
            # Shapes of x_s and x_d here: [N, C, H // 2, W]
            x = torch.cat([x_s, x_d], dim=2)
        else:  # data_format == NHWC_FORMAT:
            # Shapes of x_s and x_d here: [N, H // 2, W, C]
            x = torch.cat([x_s, x_d], dim=1)
    return x


def join_coeffs_after_inv_1d_op(coeffs, src_shape, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
    assert not (across_cols and across_rows) and (across_cols or across_rows)
    x_ev_0, x_od_0 = coeffs
    # Thanks to https://stackoverflow.com/questions/44952886/tensorflow-merge-two-2-d-tensors-according-to-even-and-odd-indices,
    # answer by P-Gn
    if data_format == NCHW_FORMAT:
        _, C, H, W = src_shape
        if across_cols:
            # x_od_0 shape: [N, C, H, W // 2], x_ev_0 shape: [N, C, H, W // 2] -> [N, C, H, W // 2, 2] -> [N, C, H, W]
            x = torch.stack([x_ev_0, x_od_0], dim=4)
            x = torch.reshape(x, [-1, C, H, 1 * W])
        else: # across_rows:
            # x_od_0 shape: [N, C, H // 2, W], x_ev_0 shape: [N, C, H // 2, W] -> [N, C, H // 2, 2, W] -> [N, C, H, W]
            x = torch.stack([x_ev_0, x_od_0], dim=3)
            x = torch.reshape(x, [-1, C, 1 * H, W])
    else:  # data_format == NHWC_FORMAT:
        # Axis is the next after spatial dim
        _, H, W, C = src_shape
        if across_cols:
            # x_od_0 shape: [N, H, W // 2, C], x_ev_0 shape: [N, H, W // 2, C] -> [N, H, W // 2, 2, C] -> [N, H, W, C]
            x = torch.stack([x_ev_0, x_od_0], dim=3)
            x = torch.reshape(x, [-1, H, 1 * W, C])
        else: # across_rows:
            # x_od_0 shape: [N, H // 2, W, C], x_ev_0 shape: [N, H // 2, W, C] -> [N, H // 2, 2, W, C] -> [N, H, W, C]
            x = torch.stack([x_ev_0, x_od_0], dim=2)
            x = torch.reshape(x, [-1, 1 * H, W, C])
    return x


def merge_coeffs_into_channels(x_coeffs, data_format=DEFAULT_DATA_FORMAT):
    x_LL, x_LH, x_HL, x_HH = x_coeffs
    if data_format == NCHW_FORMAT:
        concat_axis = 1
    else: # if data_format == NHWC_FORMAT:
        concat_axis = 3
    return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=concat_axis)


def extract_coeffs_from_channels(x, data_format=DEFAULT_DATA_FORMAT):
    if data_format == NCHW_FORMAT:
        n = x.shape[1] // 4
        x_LL = x[:, (0 * n) : (1 * n), :, :]
        x_LH = x[:, (1 * n) : (2 * n), :, :]
        x_HL = x[:, (2 * n) : (3 * n), :, :]
        x_HH = x[:, (3 * n) : (4 * n), :, :]
    else: # if data_format == NHWC_FORMAT:
        n = x.shape[3] // 4
        x_LL = x[:, :, :, (0 * n) : (1 * n)]
        x_LH = x[:, : ,:, (1 * n) : (2 * n)]
        x_HL = x[:, :, :, (2 * n) : (3 * n)]
        x_HH = x[:, :, :, (3 * n) : (4 * n)]
    return x_LL, x_LH, x_HL, x_HH


def merge_coeffs_into_spatial(x_coeffs, data_format=DEFAULT_DATA_FORMAT):
    x_LL, x_LH, x_HL, x_HH = x_coeffs
    if data_format == NCHW_FORMAT:
        h_axis, v_axis = 2, 3
    else:  # if data_format == NHWC_FORMAT:
        h_axis, v_axis = 1, 2
    x = torch.cat([
        torch.cat([x_LL, x_LH], dim=h_axis),
        torch.cat([x_HL, x_HH], dim=h_axis)
    ], dim=v_axis)
    return x


def extract_coeffs_from_spatial(x, data_format=DEFAULT_DATA_FORMAT):
    if data_format == NCHW_FORMAT:
        _, C, H, W = x.shape
        x_LL = x[:, :, : H // 2, : W // 2]
        x_LH = x[:, :, H // 2 :, : W // 2]
        x_HL = x[:, :, : H // 2, W // 2: ]
        x_HH = x[:, :, H // 2 :, W // 2: ]
    else:  # data_format == NHWC_FORMAT:
        _, H, W, C = x.shape
        x_LL = x[:, : H // 2, : W // 2, :]
        x_LH = x[:, H // 2 :, : W // 2, :]
        x_HL = x[:, : H // 2, W // 2 :, :]
        x_HH = x[:, H // 2 :, W // 2 :, :]
    return x_LL, x_LH, x_HL, x_HH


# ----- New vectorized versions -----

def convert_paddings(pads):
    # Reverse pads from [dim1, dim2, ... dimN] to [dimN, ..., dim2, dim1] and then flatten
    new_pads = [pad_value for dim_pad in reversed(pads) for pad_value in dim_pad]
    return new_pads


def pos_shift_4d(x, n_shifts, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
    # x shape: [N, C, H, W] or [N, H, W, C]
    # x[i] -> x[i + n], so remove the first n elements and pad on the right side
    assert not (across_cols and across_rows) and (across_cols or across_rows)
    if data_format == NCHW_FORMAT:
        if across_cols:
            # Inputs have shape NCHW and operation is applied across W axis (cols)
            paddings = convert_paddings([[0, 0], [0, 0], [0, 0], [0, n_shifts]])
            padded_x = F.pad(x[:, :, :, n_shifts:], paddings, mode=PAD_MODE)
        else: # across_rows:
            # Inputs have shape NCHW and operation is applied across H axis (rows)
            paddings = convert_paddings([[0,0], [0, 0], [0, n_shifts], [0, 0]])
            padded_x = F.pad(x[:, :, n_shifts:, :], paddings, mode=PAD_MODE)
    else:  # data_format == NHWC_FORMAT:
        if across_cols:
            # Inputs have shape NHWC and operation is applied across W axis (cols)
            paddings = convert_paddings([[0, 0], [0, 0], [0, n_shifts], [0, 0]])
            padded_x = F.pad(x[:, :, n_shifts:, :], paddings, mode=PAD_MODE)
        else: # across_rows:
            # Inputs have shape NHWC and operation is applied across H axis (rows)
            paddings = convert_paddings([[0, 0], [0, n_shifts], [0, 0], [0, 0]])
            padded_x = F.pad(x[:, n_shifts:, :, :], paddings, mode=PAD_MODE)
    return padded_x


def neg_shift_4d(x, n_shifts, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
    # x shape: [N, C, H, W] or [N, H, W, C]
    # x[i] -> x[i - n], so remove the last n elements and pad on the left side
    assert not(across_cols and across_rows) and (across_cols or across_rows)
    if data_format == NCHW_FORMAT:
        if across_cols:
            # Inputs have shape NCHW and operation is applied across W axis (cols)
            paddings = convert_paddings([[0, 0], [0, 0], [0, 0], [n_shifts, 0]])
            padded_x = F.pad(x[:, :, :, :-n_shifts], paddings, mode=PAD_MODE)
        else: # across_rows:
            # Inputs have shape NCHW and operation is applied across H axis (rows)
            paddings = convert_paddings([[0, 0], [0, 0], [n_shifts, 0], [0, 0]])
            padded_x = F.pad(x[:, :, :-n_shifts, :], paddings, mode=PAD_MODE)
    else:  # data_format == NHWC_FORMAT:
        if across_cols:
            # Inputs have shape NHWC and operation is applied across W axis (cols)
            paddings = convert_paddings([[0, 0], [0, 0], [n_shifts, 0], [0, 0]])
            padded_x = F.pad(x[:, :, :-n_shifts, :], paddings, mode=PAD_MODE)
        else: # across_rows:
            # Inputs have shape NHWC and operation is applied across H axis (rows)
            paddings = convert_paddings([[0, 0], [n_shifts, 0], [0, 0], [0, 0]])
            padded_x = F.pad(x[:, :-n_shifts, :, :], paddings, mode=PAD_MODE)
    return padded_x
