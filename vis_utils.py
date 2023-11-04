import os
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_grayscale_fname(path):
    ext = path.rsplit('.', 1)[1]
    new_image_path = path.rsplit('.', 1)[0] + f'_grayscale.{ext}'
    return new_image_path


def prepare_input_image(path_idx=0):
    image_names = [
        'LSUN_car_001113_256x256.png',
        'LSUN_car_000970_384x512.jpg',
        'castle1.png',
        'castle2.png'
    ]
    add_grayscale_paths = True
    if add_grayscale_paths:
        image_names += [to_grayscale_fname(x) for x in image_names]
    image_names += ['textures.png']

    path_prefix = os.path.dirname(os.path.realpath(__file__))
    image_path = os.path.join(path_prefix, 'images', image_names[path_idx])
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_size_yx = image.shape[:2]
    max_size = 512
    if min(image_size_yx) > max_size:
        scale = max(image_size_yx[0] / max_size, image_size_yx[1] / max_size)
        image = cv2.resize(image, (None), fx=1./scale, fy=1./scale, interpolation=cv2.INTER_AREA)

    # Do it only after resize
    if len(image.shape) == 2:
        # Add channel for grayscale images
        image = image[..., None]

    use_max_size = False
    if use_max_size:
        max_size = 256
        image = image[:max_size, :max_size]
        image_h, image_w = image.shape[:2]
        image = np.pad(image, [[0, max_size - image_h], [0, max_size - image_w], [0, 0]])
    return image, image_path


def show_lifting_results(src_image, anz_image, wavelet_name):
    plt.figure(1)
    plt.imshow(src_image[..., ::-1])
    plt.title('Src input image')
    plt.figure(2)
    plt.imshow(anz_image[..., ::-1])
    plt.title(f'Anz image with {wavelet_name}')
    plt.show()


def add_title_to_image(image, title):
    # For centering thanks to: https://gist.github.com/evilmtv/af2a023e472e6303fd2d3cc02aa1a83a
    h, w, c = image.shape[:3]
    title_h = int(0.1 * h)
    title_image = np.zeros((title_h, w, c), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = h / 384 # 1 for image size 384 is fine
    font_thickness = 2
    text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
    text_x = int(w / 2 - (text_size[0] / 2))
    text_y = int(title_h / 2 + (text_size[1] / 2))
    title_image = cv2.putText(title_image, title, (text_x, text_y), font,
                              font_scale, (255, 255, 255), font_thickness)

    return np.vstack([title_image, image])


def create_images_grid(images, n_cols, n_rows):
    h, w, c = images[0].shape
    grid_h = h * n_rows
    grid_w = w * n_cols
    grid_image = np.zeros((grid_h, grid_w, c), dtype=np.uint8)
    for row in range(n_rows):
        for col in range(n_cols):
            x_start = w * col
            x_end = x_start + w
            y_start = h * row
            y_end = y_start + h
            image_idx = (row * n_cols) + col
            if image_idx < len(images):
                grid_image[y_start : y_end, x_start : x_end] = images[image_idx]
    return grid_image


def prepare_textures_image():
    image_size = 512
    h = w = image_size
    image = np.zeros((h, w), dtype=np.uint8)
    thickness = 10
    color = 1
    triangle_points = np.array([
        [64 + 16, 32], [32, 128], [128, 128]
    ])
    cross_points = np.array([
        [32, 160], [128, 256], [128, 160], [32, 256]
    ])
    square_points = np.array([
        [160, 160], [256, 160], [256, 256], [160, 256]
    ])
    circle_center, circle_r = (192 + 16, 64 + 16), 32 + 16
    image = cv2.polylines(image, [triangle_points.reshape(1, -1, 2)], True, color , thickness=thickness)
    image = cv2.polylines(image, [cross_points.reshape(1, -1, 2)], True, color, thickness=thickness)
    image = cv2.polylines(image, [square_points.reshape(1, -1, 2)], True, color, thickness=thickness)
    image = cv2.circle(image, circle_center, circle_r, color, thickness=thickness)
    lines = [
        # 3 grouped lines
        [[32, 352], [96, 288]],
        [[64, 352], [160, 288]],
        [[96, 352], [256, 288]],
        # 3 grouped lines
        [[288, 32], [352, 96]],
        [[288, 64], [352, 160]],
        [[288, 128], [352, 256]],
        # Cross 1
        [[288, 288], [480, 480]],
        [[480, 288], [288, 480]],
        # Cross 2
        [[384, 32], [480, 256]],
        [[384, 256], [480, 32]],
        # Cross 3
        [[32, 384], [256, 480]],
        [[32, 480], [256, 384]],
        # Bottom right cross additional lines
        [[288, 320], [288, 448]],
        [[480, 320], [480, 448]],
        [[320, 480], [448, 480]],
        [[320, 288], [448, 288]]
    ]
    for line_points in lines:
        image = cv2.polylines(image, [np.array(line_points).reshape(1, -1, 2)], True, color, thickness=thickness)

    # Convert to bgr in range [0, 255]
    # image = cv2.cvtColor((255 * image), cv2.COLOR_GRAY2RGB)
    image = 255 * image

    fig, ax = plt.subplots()
    ticks = np.arange(0, image_size, image_size // 16)
    ax.imshow(image)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid('--', which='both', alpha=0.5)
    plt.show()
    image_path = os.path.join('images', 'textures.png')
    cv2.imwrite(image_path, image)
    print(f'Saved texture image at {image_path}')
    return image


def prepare_grayscale_images():
    images_paths = [p for p in glob(os.path.join('images', '*')) if os.path.isfile(p)]
    for image_path in images_paths:
        name = os.path.basename(image_path)
        if 'wavelets' not in name.lower():
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                new_image_path = to_grayscale_fname(image_path)
                cv2.imwrite(new_image_path, image)
                print(f'Created grayscale image at {new_image_path}')


if __name__ == '__main__':
    # prepare_textures_image()
    prepare_grayscale_images()
