import numpy as np
import torch
import torch.nn.functional as F


NCHW_FORMAT = 'NCHW'
NHWC_FORMAT = 'NHWC'
to_NHWC_axis = [0, 2, 3, 1] # NCHW -> NHWC
to_NCHW_axis = [0, 3, 1, 2] # NHWC -> NCHW

# Note: in PyTorch NCHW and NHWC are the same as for indexes, only strides differ, so always use NCHW
DEFAULT_DATA_FORMAT = NCHW_FORMAT

PAD_MODE = 'constant'

def get_default_coeffs_scales_2d(COEFFS_SCALES_V):
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

    COEFFS_SCALES_2D = torch.from_numpy(COEFFS_SCALES_2D_DICT[COEFFS_SCALES_V])
    return COEFFS_SCALES_2D

# 6 is the best for preserving source data range for LL and keeping similar ranges for all H details
# Found with tests.py with and without normalization for input
COEFFS_SCALES_V = 6
COEFFS_SCALES_2D = get_default_coeffs_scales_2d(COEFFS_SCALES_V)

DEFAULT_SCALE_1D_COEFFS = True
DEFAULT_SCALE_2D_COEFFS = True

# Scales for LL, LH, HL, HH after DWT. Before IDWT inverse values must be used
# Note: probably it's better to combine scaled init with these coeffs.
# In this case coeffs can be scaled by corresponding values, e.g.,
# layer_scales = [2, 16, 16, 24] and weight_init_scales = [1, 8, 8, 8]
LAYER_COEFFS_SCALES = [2, 48, 48, 64]


# ----- 1d op utils -----

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


# ----- Merge/extract utils -----

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


# ----- Shifts with paddings -----

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
