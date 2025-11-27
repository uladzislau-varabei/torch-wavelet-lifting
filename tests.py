import os
import time

import cv2
import pandas as pd

from wavelets.utils import COEFFS_SCALES_V, get_default_coeffs_scales_2d
from wavelets.test_utils import test_lifting_scheme, test_lifting_scales
from wavelets import WAVELETS_LIST
from vis_utils import prepare_input_image, add_title_to_image, create_images_grid


def test_wavelets(save_image, coeffs_scales_v):
    print('----- Testing wavelets -----')
    image_path_idx = 8
    image, image_path = prepare_input_image(path_idx=image_path_idx)
    image_name = os.path.split(image_path)[1].rsplit('.', 1)[0]
    grid_n_cols = 5
    grid_n_rows = (len(WAVELETS_LIST) // grid_n_cols) + (1 if len(WAVELETS_LIST) % grid_n_cols else 0)
    grid_images = []
    grid_title = 'Wavelets test'
    grid_path = os.path.join('results', f'wavelets_{image_name}.png')
    os.makedirs(os.path.dirname(grid_path), exist_ok=True)
    scales_2d = get_default_coeffs_scales_2d(coeffs_scales_v)
    stats_csv_path = os.path.join('stats', f'wavelets_stats_scales-v{coeffs_scales_v}_{image_name}.csv')
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
    for idx, (k, v) in enumerate(errors.items(), 1):
        print(f'{str(idx):>2s}) {k:<21s}: {v}')
    # Process scales stats
    print('Scales stats:')
    scales_stats_df_info = []
    for idx, (k, v) in enumerate(scales_stats.items(), 1):
        scales_stats_df_info.append({'name': k, **v})
        print(f'{str(idx):>2s}) {k:<21s}: {v}')
    scales_stats_df = pd.DataFrame(scales_stats_df_info)
    scales_stats_df.to_csv(stats_csv_path, index=False, sep=';')
    print(f'Saved stats to: {stats_csv_path}')
    print('----- Finished wavelets testing -----')


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
    coeffs_scales_v = COEFFS_SCALES_V
    test_wavelets(save_image, coeffs_scales_v)

    wavelet_name = 'cdf-9/7'
    # wavelet_name = 'haar'
    # wavelet_name = 'cdf-5/3'
    normalize_input = True
    plot_data = True
    plot_hist = False
    # test_scales(wavelet_name, normalize_input, plot_data, plot_hist)
