import os
import time

import cv2

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
from wavelets.utils import test_lifting_scheme
from vis_utils import prepare_input_image, add_title_to_image, create_images_grid


if __name__ == '__main__':
    # Use 4 cols
    wavelets_list = [
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

    image_path_idx = 8
    image, image_path = prepare_input_image(path_idx=image_path_idx)
    image_name = os.path.split(image_path)[1].rsplit('.', 1)[0]
    grid_n_cols = 5
    grid_n_rows = (len(wavelets_list) // grid_n_cols) + (1 if len(wavelets_list) % grid_n_cols else 0)
    grid_images = []
    grid_title = 'Wavelets test'
    grid_path = os.path.join('results', f'wavelets_{image_name}.png')
    errors = {}
    start_time = time.time()
    for wavelet in wavelets_list:
        name, forward_op, backward_op, kernel = wavelet
        anz_image, error = test_lifting_scheme(image=image.copy(),
                                               kernel=kernel,
                                               forward_2d_op=forward_op,
                                               backward_2d_op=backward_op,
                                               print_logs=False)
        anz_image = add_title_to_image(anz_image, name)
        grid_images.append(anz_image)
        errors[name] = error
    total_time = time.time() - start_time
    grid_image = create_images_grid(grid_images, n_cols=grid_n_cols, n_rows=grid_n_rows)
    grid_image = add_title_to_image(grid_image, grid_title)
    cv2.imwrite(grid_path, grid_image)
    print(f'Processed all wavelets in {total_time:.3f} sec')
    print(f'Saved grid image to {grid_path}')
    print('Reconstruction errors:')
    for idx, (k, v) in enumerate(errors.items()):
        print(f'{idx + 1}) {k}: {v}')
