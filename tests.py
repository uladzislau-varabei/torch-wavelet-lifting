import os
import time

import cv2
import pandas as pd

from wavelets.bior_spline_33 import (
    fast_biorspline33_2d_op, fast_inv_biorspline33_2d_op, BIOR_SPLINE_33_KERNEL
)
from wavelets.bior_spline_35 import (
    fast_biorspline35_2d_op, fast_inv_biorspline35_2d_op, BIOR_SPLINE_35_KERNEL
)
from wavelets.bior_spline_37 import (
    fast_biorspline37_2d_op, fast_inv_biorspline37_2d_op, BIOR_SPLINE_37_KERNEL
)
from wavelets.bior_spline_39 import (
    fast_biorspline39_2d_op, fast_inv_biorspline39_2d_op, BIOR_SPLINE_39_KERNEL
)
from wavelets.bior_spline_48 import (
    fast_biorspline48_2d_op, fast_inv_biorspline48_2d_op, BIOR_SPLINE_48_KERNEL
)
from wavelets.rev_bior_spline_33 import (
    fast_revbiorspline33_2d_op, fast_inv_revbiorspline33_2d_op, REV_BIOR_SPLINE_33_KERNEL
)
from wavelets.rev_bior_spline_35 import (
    fast_revbiorspline35_2d_op, fast_inv_revbiorspline35_2d_op, REV_BIOR_SPLINE_35_KERNEL
)
from wavelets.rev_bior_spline_37 import (
    fast_revbiorspline37_2d_op, fast_inv_revbiorspline37_2d_op, REV_BIOR_SPLINE_37_KERNEL
)
from wavelets.rev_bior_spline_39 import (
    fast_revbiorspline39_2d_op, fast_inv_revbiorspline39_2d_op, REV_BIOR_SPLINE_39_KERNEL
)
from wavelets.rev_bior_spline_48 import (
    fast_revbiorspline48_2d_op, fast_inv_revbiorspline48_2d_op, REV_BIOR_SPLINE_48_KERNEL
)
from wavelets.cdf_53 import fast_cdf53_2d_op, fast_inv_cdf53_2d_op, CDF_53_KERNEL
from wavelets.cdf_97 import fast_cdf97_2d_op, fast_inv_cdf97_2d_op, CDF_97_KERNEL
from wavelets.haar import fast_haar_2d_op, fast_inv_haar_2d_op, HAAR_KERNEL
from wavelets.daub_4 import fast_daub4_2d_op, fast_inv_daub4_2d_op, DAUB4_KERNEL
from wavelets.coif_12 import fast_coif12_2d_op, fast_inv_coif12_2d_op, COIF12_KERNEL
from wavelets.utils import test_lifting_scheme, test_lifting_scales, COEFFS_SCALES_V
from vis_utils import prepare_input_image, add_title_to_image, create_images_grid


WAVELETS_LIST = [
    ['CDF-9/7', fast_cdf97_2d_op, fast_inv_cdf97_2d_op, CDF_97_KERNEL],
    ['CDF-5/3', fast_cdf53_2d_op, fast_inv_cdf53_2d_op, CDF_53_KERNEL],
    ['Haar', fast_haar_2d_op, fast_inv_haar_2d_op, HAAR_KERNEL],
    ['Daubechies-4', fast_daub4_2d_op, fast_inv_daub4_2d_op, DAUB4_KERNEL],
    ['Coiflet-12', fast_coif12_2d_op, fast_inv_coif12_2d_op, COIF12_KERNEL],
    ['Bior_spline-3/3', fast_biorspline33_2d_op, fast_inv_biorspline33_2d_op, BIOR_SPLINE_33_KERNEL],
    ['Bior_spline-3/5', fast_biorspline35_2d_op, fast_inv_biorspline35_2d_op, BIOR_SPLINE_35_KERNEL],
    ['Bior_spline-3/7', fast_biorspline37_2d_op, fast_inv_biorspline37_2d_op, BIOR_SPLINE_37_KERNEL],
    ['Bior_spline-3/9', fast_biorspline39_2d_op, fast_inv_biorspline39_2d_op, BIOR_SPLINE_39_KERNEL],
    ['Bior_spline-4/8', fast_biorspline48_2d_op, fast_inv_biorspline48_2d_op, BIOR_SPLINE_48_KERNEL],
    ['Rev_bior_spline-3/3', fast_revbiorspline33_2d_op, fast_inv_revbiorspline33_2d_op, REV_BIOR_SPLINE_33_KERNEL],
    ['Rev_bior_spline-3/5', fast_revbiorspline35_2d_op, fast_inv_revbiorspline35_2d_op, REV_BIOR_SPLINE_35_KERNEL],
    ['Rev_bior_spline-3/7', fast_revbiorspline37_2d_op, fast_inv_revbiorspline37_2d_op, REV_BIOR_SPLINE_37_KERNEL],
    ['Rev_bior_spline-3/9', fast_revbiorspline39_2d_op, fast_inv_revbiorspline39_2d_op, REV_BIOR_SPLINE_39_KERNEL],
    ['Rev_bior_spline-4/8', fast_revbiorspline48_2d_op, fast_inv_revbiorspline48_2d_op, REV_BIOR_SPLINE_48_KERNEL],
]


def test_wavelets(save_image):
    image_path_idx = 8
    image, image_path = prepare_input_image(path_idx=image_path_idx)
    image_name = os.path.split(image_path)[1].rsplit('.', 1)[0]
    grid_n_cols = 5
    grid_n_rows = (len(WAVELETS_LIST) // grid_n_cols) + (1 if len(WAVELETS_LIST) % grid_n_cols else 0)
    grid_images = []
    grid_title = 'Wavelets test'
    grid_path = os.path.join('results', f'wavelets_{image_name}.png')
    os.makedirs(os.path.dirname(grid_path), exist_ok=True)
    stats_csv_path = os.path.join('stats', f'wavelets_stats_scales-{COEFFS_SCALES_V}_{image_name}.csv')
    os.makedirs(os.path.dirname(stats_csv_path), exist_ok=True)
    errors = {}
    scales_stats = {}
    start_time = time.time()
    for wavelet in WAVELETS_LIST:
        name, forward_op, backward_op, kernel = wavelet
        anz_image, error, scales = test_lifting_scheme(image=image.copy(),
                                                       kernel=kernel,
                                                       forward_2d_op=forward_op,
                                                       backward_2d_op=backward_op,
                                                       print_logs=False)
        anz_image = add_title_to_image(anz_image, name)
        grid_images.append(anz_image)
        errors[name] = error
        scales_stats[name] = scales
    total_time = time.time() - start_time
    print(f'Processed all wavelets in {total_time:.3f} sec')
    # Optionally save image grid
    if save_image:
        grid_image = create_images_grid(grid_images, n_cols=grid_n_cols, n_rows=grid_n_rows)
        grid_image = add_title_to_image(grid_image, grid_title)
        cv2.imwrite(grid_path, grid_image)
        print(f'Saved grid image to {grid_path}')
    else:
        print(f'Image is not saved')
    # Process errors stats
    print('Reconstruction errors:')
    for idx, (k, v) in enumerate(errors.items()):
        print(f'{idx + 1}) {k}: {v}')
    # Process scales stats
    print('Scales stats:')
    scales_stats_df_info = []
    for idx, (k, v) in enumerate(scales_stats.items()):
        scales_stats_df_info.append({'name': k, **v})
        print(f'{idx + 1}) {k}: {v}')
    scales_stats_df = pd.DataFrame(scales_stats_df_info)
    scales_stats_df.to_csv(stats_csv_path, index=False, sep=';')
    print(f'Saved stats to: {stats_csv_path}')


def test_scales(wavelet_name, normalize_input, plot_data, plot_hist):
    image_path_idx = 6
    image, image_path = prepare_input_image(path_idx=image_path_idx)
    wavelet = None
    for w in WAVELETS_LIST:
        if w[0].lower() == wavelet_name.lower():
            wavelet = w
            break
    assert wavelet is not None, f'wavelet_name={wavelet_name} is not supported'
    test_lifting_scales(image, name=wavelet[0], kernel=wavelet[3], forward_2d_op=wavelet[1],
                        normalize_input=normalize_input,
                        plot_data=plot_data, plot_hist=plot_hist)


if __name__ == '__main__':
    save_image = False
    test_wavelets(save_image)

    wavelet_name = 'cdf-9/7'
    # wavelet_name = 'haar'
    # wavelet_name = 'cdf-5/3'
    normalize_input = True
    plot_data = True
    plot_hist = False
    test_scales(wavelet_name, normalize_input, plot_data, plot_hist)
