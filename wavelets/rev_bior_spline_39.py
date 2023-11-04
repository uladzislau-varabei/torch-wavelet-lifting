import numpy as np
import matplotlib.pyplot as plt

from wavelets.utils import NCHW_FORMAT, NHWC_FORMAT, DEFAULT_DATA_FORMAT, COEFFS_SCALES_2D, \
    prepare_coeffs_for_1d_op, prepare_coeffs_for_inv_1d_op, \
    extract_coeffs_from_channels, extract_coeffs_from_spatial, \
    merge_coeffs_into_channels, merge_coeffs_into_spatial, \
    pos_shift_4d, neg_shift_4d, \
    join_coeffs_after_1d_op, join_coeffs_after_inv_1d_op, \
    test_lifting_scheme
from vis_utils import prepare_input_image, show_lifting_results


d1 = 1. / 3 # step 1 for [i+1]
d2 = -7 / 8192 # step 2 for [i-4]
d3 = 329. / 36864 # step 2 for [i-3]
d4 = -547. / 12288 # step 2 for [i-2]
d5 = 1831. / 12288 # step 2 for [i-1]
d6 = -4. / 9 # step 2 for [i]
d7 = -1831. / 12288 # step 2 for [i+1]
d8 = 547. / 12288 # step 2 for [i+2]
d9 = -329 / 36864 # step 2 for [i+3]
d10 = 7. / 8192 # step 2 for [i+4]
c1 = 3. / 8 # step 1 for [i-1]
c2 = 9. / 8 # step 1 for [i]
k = np.sqrt(2.) / 3
DEFAULT_KERNEL = [c1, c2, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, k]
REV_BIOR_SPLINE_39_KERNEL = DEFAULT_KERNEL


# ----- New vectorized versions -----

def fast_revbiorspline39_1d_op(x, kernel, scale_coeffs, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
    # x shape: [N, C, H, W] or [N, H, W, C]
    assert not(across_cols and across_rows) and (across_cols or across_rows), \
        f'Reverse-Bior-Spline-3/9 1d op: across_cols = {across_cols}, across_rows = {across_rows}'
    common_kwargs = {
        'across_cols': across_cols,
        'across_rows': across_rows,
        'data_format': data_format
    }
    # Split coeffs
    x_ev_0, x_od_0 = prepare_coeffs_for_1d_op(x, **common_kwargs)
    # o - odd, e - even
    c1, c2, k = kernel[0], kernel[1], kernel[12]
    d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 =\
        kernel[2], kernel[3], kernel[4], kernel[5], kernel[6], \
            kernel[7], kernel[8], kernel[9], kernel[10], kernel[11]
    x_od_1 = x_od_0 + (
        d1 * pos_shift_4d(x_ev_0, n_shifts=1, **common_kwargs)
    )
    x_ev_1 = x_ev_0 + (
        c1 * neg_shift_4d(x_od_1, n_shifts=1, **common_kwargs) +
        c2 * x_od_1
    )
    x_od_2 = x_od_1 + (
        d2  * neg_shift_4d(x_ev_1, n_shifts=4, **common_kwargs) +
        d3  * neg_shift_4d(x_ev_1, n_shifts=3, **common_kwargs) +
        d4  * neg_shift_4d(x_ev_1, n_shifts=2, **common_kwargs) +
        d5  * neg_shift_4d(x_ev_1, n_shifts=1, **common_kwargs) +
        d6  * x_ev_1 +
        d7  * pos_shift_4d(x_ev_1, n_shifts=1, **common_kwargs) +
        d8  * pos_shift_4d(x_ev_1, n_shifts=2, **common_kwargs) +
        d9  * pos_shift_4d(x_ev_1, n_shifts=3, **common_kwargs) +
        d10 * pos_shift_4d(x_ev_1, n_shifts=4, **common_kwargs)
    )
    # Normalization
    if scale_coeffs:
        x_ev_2 = k * x_ev_1 # s
        x_od_3 = (1. / k) * x_od_2 # d
    else:
        x_ev_2, x_od_3 = x_ev_1, x_od_2 # s, d
    # Join coeffs
    x = join_coeffs_after_1d_op((x_ev_2, x_od_3), **common_kwargs)
    return x


def fast_inv_revbiorspline39_1d_op(x_coefs, kernel, scale_coeffs, across_cols=False, across_rows=False, data_format=DEFAULT_DATA_FORMAT):
    # x shape: [N, C, H, W] or [N, H, W, C]
    assert not(across_cols and across_rows) and (across_cols or across_rows), \
        f'Reverse-Bior-Spline-3/9 1d op: across_cols = {across_cols}, across_rows = {across_rows}'
    common_kwargs = {
        'across_cols': across_cols,
        'across_rows': across_rows,
        'data_format': data_format
    }
    # x_coefs: s, d
    s, d = prepare_coeffs_for_inv_1d_op(x_coefs, across_cols=across_cols, across_rows=across_rows, data_format=data_format)
    # o - odd, e - even
    c1, c2, k = kernel[0], kernel[1], kernel[12]
    d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 =\
        kernel[2], kernel[3], kernel[4], kernel[5], kernel[6], \
            kernel[7], kernel[8], kernel[9], kernel[10], kernel[11]
    if scale_coeffs:
        x_ev_1 = (1. / k) * s
        x_od_2 = k * d
    else:
        x_ev_1, x_od_2 = s, d
    x_od_1 = x_od_2 - (
        d2  * neg_shift_4d(x_ev_1, n_shifts=4, **common_kwargs) +
        d3  * neg_shift_4d(x_ev_1, n_shifts=3, **common_kwargs) +
        d4  * neg_shift_4d(x_ev_1, n_shifts=2, **common_kwargs) +
        d5  * neg_shift_4d(x_ev_1, n_shifts=1, **common_kwargs) +
        d6  * x_ev_1 +
        d7  * pos_shift_4d(x_ev_1, n_shifts=1, **common_kwargs) +
        d8  * pos_shift_4d(x_ev_1, n_shifts=2, **common_kwargs) +
        d9  * pos_shift_4d(x_ev_1, n_shifts=3, **common_kwargs) +
        d10 * pos_shift_4d(x_ev_1, n_shifts=4, **common_kwargs)
    )
    x_ev_0 = x_ev_1 - (
        c1 * neg_shift_4d(x_od_1, n_shifts=1, **common_kwargs) +
        c2 * x_od_1
    )
    x_od_0 = x_od_1 - (
        d1 * pos_shift_4d(x_ev_0, n_shifts=1, **common_kwargs)
    )
    # Join coeffs
    x = join_coeffs_after_inv_1d_op((x_ev_0, x_od_0), src_shape=x_coefs.shape, **common_kwargs)
    return x


#@tf.function(jit_compile=True)
def fast_revbiorspline39_2d_op(x, kernel, scale_1d_coeffs, scale_2d_coeffs, coeffs_scales_2d, data_format=DEFAULT_DATA_FORMAT):
    # 1. Apply across rows
    x = fast_revbiorspline39_1d_op(x, kernel, scale_coeffs=scale_1d_coeffs, across_cols=False, across_rows=True, data_format=data_format)
    # 2. Apply across cols
    x = fast_revbiorspline39_1d_op(x, kernel, scale_coeffs=scale_1d_coeffs, across_cols=True, across_rows=False, data_format=data_format)
    # 3. Rearrange images from spatial into channels
    x_LL, x_LH, x_HL, x_HH = extract_coeffs_from_spatial(x, data_format=data_format)
    if scale_2d_coeffs:
        coeffs_scales = coeffs_scales_2d
        x_LL *= coeffs_scales[0]
        x_LH *= coeffs_scales[1]
        x_HL *= coeffs_scales[2]
        x_HH *= coeffs_scales[3]
    x_output = merge_coeffs_into_channels([x_LL, x_LH, x_HL, x_HH], data_format=data_format)
    return x_output


def fast_inv_revbiorspline39_2d_op(x, kernel, scale_1d_coeffs, scale_2d_coeffs, coeffs_scales_2d, data_format=DEFAULT_DATA_FORMAT):
    # x_LL, x_LH, x_HL, x_HH = x_coeffs
    # 1. Rearrange images from channels into spatial
    x_LL, x_LH, x_HL, x_HH = extract_coeffs_from_channels(x, data_format=data_format)
    if scale_2d_coeffs:
        coeffs_scales = 1. / coeffs_scales_2d
        x_LL *= coeffs_scales[0]
        x_LH *= coeffs_scales[1]
        x_HL *= coeffs_scales[2]
        x_HH *= coeffs_scales[3]
    x = merge_coeffs_into_spatial([x_LL, x_LH, x_HL, x_HH], data_format=data_format)
    # 2. Apply inverse transform across cols
    x = fast_inv_revbiorspline39_1d_op(x, kernel, scale_coeffs=scale_1d_coeffs, across_cols=True, across_rows=False, data_format=data_format)
    # 3. Apply inverse transform across rows
    x = fast_inv_revbiorspline39_1d_op(x, kernel, scale_coeffs=scale_1d_coeffs, across_cols=False, across_rows=True, data_format=data_format)
    return x


# ----- Main -----

if __name__ == '__main__':
    image, _ = prepare_input_image()
    data_format = NHWC_FORMAT
    #data_format = NCHW_FORMAT
    vis_anz_image, error = test_lifting_scheme(image,
                                               kernel=DEFAULT_KERNEL,
                                               forward_2d_op=fast_revbiorspline39_2d_op,
                                               backward_2d_op=fast_inv_revbiorspline39_2d_op,
                                               data_format=data_format)

    show_lifting_results(src_image=image, anz_image=vis_anz_image, wavelet_name='Reverse Bior Spline 3/9')
