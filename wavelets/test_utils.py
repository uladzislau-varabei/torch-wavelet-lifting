import copy

import numpy as np
import pandas as pd
import torch

from wavelets.utils import NCHW_FORMAT, DEFAULT_DATA_FORMAT, COEFFS_SCALES_2D, \
    get_default_coeffs_scales_2d, extract_coeffs_from_channels, merge_coeffs_into_spatial


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


def preprocess_image(image, data_format, print_logs=False):
    if data_format == NCHW_FORMAT:
        image = np.transpose(image, (2, 0, 1))

    input_image = image[None, ...].astype(np.float32)
    input_image = (input_image / 127.5) - 1.0
    if print_logs:
        print(f'Input image min: {input_image.min()}, max: {input_image.max()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_image = torch.from_numpy(input_image).to(device)
    return input_image


def test_lifting_scheme(image, kernel, forward_2d_op, backward_2d_op, scale_1d_coefs=True, scale_2d_coefs=True,
                        data_format=DEFAULT_DATA_FORMAT, print_logs=True):
    input_image = preprocess_image(image, data_format=data_format, print_logs=print_logs)
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
    #for scales_v in sorted(list(COEFFS_SCALES_2D_DICT.keys())):
    for scales_v in [1, 2, 3, 4, 5, 6]:
        # coeffs_scales_2d = torch.from_numpy(COEFFS_SCALES_2D_DICT[scales_v])
        coeffs_scales_2d = torch.from_numpy(get_default_coeffs_scales_2d(scales_v))
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


def test_grad(image, kernel, forward_2d_op, backward_2d_op, scale_1d_coefs=True, scale_2d_coefs=True,
              data_format=DEFAULT_DATA_FORMAT, print_logs=True):
    if data_format == NCHW_FORMAT:
        image = np.transpose(image, (2, 0, 1))

    input_image = image[None, ...].astype(np.float32)
    input_image = (input_image / 127.5) - 1.0
    if print_logs:
        print(f'Input image min: {input_image.min()}, max: {input_image.max()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_image = (copy.deepcopy(input_image[0]) + 1) * 0.5 # scale to range [0, 1]
    input_image = torch.from_numpy(input_image).to(device)
    input_image = torch.nn.Parameter(input_image, requires_grad=True)
    coeffs_scales_2d = COEFFS_SCALES_2D

    anz_image = forward_2d_op(input_image, kernel,
                              scale_1d_coeffs=scale_1d_coefs,
                              scale_2d_coeffs=scale_2d_coefs,
                              coeffs_scales_2d=coeffs_scales_2d,
                              data_format=data_format)
    if print_logs:
        print(f'Input image shape: {input_image.shape}, anz image shape: {anz_image.shape}')
    # Apply deepcopy as in-placed ops are used and the same tensor is used later
    restored_image = backward_2d_op(anz_image, kernel,
                                    scale_1d_coeffs=scale_1d_coefs,
                                    scale_2d_coeffs=scale_2d_coefs,
                                    coeffs_scales_2d=coeffs_scales_2d,
                                    data_format=data_format)

    grads = torch.autograd.grad(outputs=[restored_image.sum()], inputs=[input_image],
        create_graph=True, only_inputs=True)[0]
    grad_diff_image = np.abs(grads[0].detach().cpu().numpy() - 1)
    print(f"Src: min={src_image.min()}, max={src_image.max()}, "
          f"grads: min={grad_diff_image.min()}, max={grad_diff_image.max()}")

    mean_grad_diff = grad_diff_image.mean()

    vis_image = np.hstack([src_image, grad_diff_image])
    return vis_image, mean_grad_diff


def find_scales_per_image_batch(images, kernel, forward_2d_op, data_format):
    def eval_base_stats(x, q, name):
        thr = 1e-8
        use_thr = True
        if use_thr:
            x_tensor = torch.abs(x.flatten())
            src_size = x_tensor.numel()
            x_tensor = x_tensor[x_tensor > thr]
            upd_size = x_tensor.numel()
            ratio = 100 * upd_size / src_size
            print(f'thr={thr}: keeping {upd_size}/{src_size} ({ratio:.1f}%) items for {name}')
        else:
            x_tensor = x.flatten()
        x_mean = x_tensor.mean().item()
        x_med = torch.median(x_tensor).item()
        assert 0 < q < 1
        # q1 = torch.quantile(x.flatten(), q).item()
        # q2 = torch.quantile(x.flatten(), 1 - q).item()
        return x_mean, x_med #, q1, q2

    def eval_params(src_stats, coeff_stats):
        mean1, min1, max1 = src_stats
        mean2, min2, max2 = coeff_stats
        scale = abs(mean1 / mean2)
        min2 = min2 * scale
        max2 = max2 * scale
        shift1 = min1 - min2
        shift2 = max1 - max2
        shift = (shift1 + shift2) / 2
        return scale, shift

    assert len(images.shape) == 4
    if data_format == NCHW_FORMAT:
        images = np.transpose(images, (0, 3, 1, 2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_images = torch.from_numpy(images).to(dtype=torch.float32, device=device)
    input_images = input_images / 127.5 - 1.0

    anz_images = forward_2d_op(input_images, kernel,
                               scale_1d_coeffs=True,
                               scale_2d_coeffs=False,
                               coeffs_scales_2d=COEFFS_SCALES_2D,
                               data_format=data_format)

    anz_coeffs = extract_coeffs_from_channels(anz_images, data_format=data_format)
    coeffs_names = ['x_LL', 'x_LH', 'x_HL', 'x_HH']

    q = 0.1
    src_stats = eval_base_stats(input_images, q, 'src')
    result = {name: eval_base_stats(c, q, name) for name, c in zip(coeffs_names, anz_coeffs)}
    result['src'] = src_stats
    return result


def find_scales(images, name, kernel, forward_2d_op, batch_size, data_format):
    """
    Approach description:
    1) Assume signal description in neural networks is in range [-1, 1] with mean 0
    2) If wavelets coeffs are of signal are predicted than initially each coeff also has range [-1, 1] with mean 0
    3) After signal is constructed from wavelets coeffs some of them have much higher impact on overall result due to transform itself
    Solution:
    1) Scale and shift each predicted coeff of signal and after that apply inverse transform
    2) What about forward transform?
    """
    from tqdm import tqdm
    results = {'x_LL': [], 'x_LH': [], 'x_HL': [], 'x_HH': [], 'src': []}
    batch_images = []
    # TODO: add support for images of different sizes
    for idx, image in tqdm(enumerate(images), total=len(images), desc=f'{name}_processing'):
        if len(batch_images) < batch_size:
            batch_images.append(image)
        if len(batch_images) == batch_size or idx == (len(images) - 1):
            batch_images = np.array(batch_images, dtype=np.uint8)
            image_scales_dict = find_scales_per_image_batch(batch_images, kernel, forward_2d_op, data_format)
            batch_images = []
            for k, v in image_scales_dict.items():
                results[k].append(v)

    stats_names = ['scale', 'shift']
    for idx in range(len(stats_names)):
        # Only process scale for now
        if idx == 0:
            stat_name = stats_names[idx]
            print(f'\n--- {name}_{stat_name} ---')
            src_array = np.array(results['src'])
            src_mean = src_array[:, idx].mean()
            src_std = src_array[:, idx].std()
            src_med = np.median(src_array[:, idx])
            print(f'src: mean={src_mean:.3f}, std={src_std:.3f}, med={src_med:.3f}')
            for k, v in results.items():
                if 'src' not in k:
                    v_array = np.array(v)[:, idx].flatten()
                    v_mean = v_array.mean()
                    v_std = v_array.std()
                    v_med = np.median(v_array)
                    scale_mean = src_mean / v_mean
                    scale_med = src_med / v_med
                    print(f'{k}: mean={v_mean:.3f}, std={v_std:.3f}, med={v_med:.3f}, '
                          f'scale_mean={scale_mean:.3f}, scale_med={scale_med:.3f}')