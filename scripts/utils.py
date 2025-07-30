import os
import yaml
import math
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def save_image_3d(tensor, slice_idx, file_name):
    '''
    tensor: [bs, c, h, w, 1]
    '''
    image_num = len(slice_idx)
    tensor = tensor[0, slice_idx, ...].permute(0, 3, 1, 2).cpu().data  # [c, 1, h, w]
    image_grid = vutils.make_grid(tensor, nrow=image_num, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, file_name, nrow=1)



def map_coordinates(input, coordinates):
    ''' PyTorch version of scipy.ndimage.interpolation.map_coordinates
    input: (B, H, W, C)
    coordinates: (2, ...)
    '''
    bs, h, w, c = input.size()

    def _coordinates_pad_wrap(h, w, coordinates):
        coordinates[0] = coordinates[0] % h
        coordinates[1] = coordinates[1] % w
        return coordinates

    co_floor = torch.floor(coordinates).long()
    co_ceil = torch.ceil(coordinates).long()
    d1 = (coordinates[1] - co_floor[1].float())
    d2 = (coordinates[0] - co_floor[0].float())
    co_floor = _coordinates_pad_wrap(h, w, co_floor)
    co_ceil = _coordinates_pad_wrap(h, w, co_ceil)

    f00 = input[:, co_floor[0], co_floor[1], :]
    f10 = input[:, co_floor[0], co_ceil[1], :]
    f01 = input[:, co_ceil[0], co_floor[1], :]
    f11 = input[:, co_ceil[0], co_ceil[1], :]
    d1 = d1[None, :, :, None].expand(bs, -1, -1, c)
    d2 = d2[None, :, :, None].expand(bs, -1, -1, c)

    fx1 = f00 + d1 * (f10 - f00)
    fx2 = f01 + d1 * (f11 - f01)
    
    return fx1 + d2 * (fx2 - fx1)


#def clear_color(x):
#    if torch.is_complex(x):
#        x = torch.abs(x)
#    x = x.detach().cpu().squeeze().numpy()
#    return normalize_np(np.transpose(x, (1, 2, 0)))
def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze(0).numpy()
    x = np.clip(x, -1, 1)
    return ((np.transpose(x, (1, 2, 0))) + 1)/2


def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img

def normalize_torch(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= torch.min(img)
    img /= torch.max(img)
    return img


"""
For inpainting:
"""

def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w


class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme', 'free_form']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                              mask_shape=(mask_h, mask_w),
                              image_size=self.image_size,
                              margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask


    def random_irregular_mask(self, img_shape,
                              num_vertices=(1, 3),  # two segments at most
                              max_angle=64,
                              length_range=(10, 40),  # length of each segment
                              brush_width=(20, 21),
                              dtype='uint8'):
        """Generate random irregular masks.
        copied from: https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/136b29f58d0af6e5db9f3655d2891f5a855fcdaa/data/util/mask.py#L232

        This is a modified version of free-form mask implemented in
        'brush_stroke_mask'.

        We prefer to use `uint8` as the data type of masks, which may be different
        from other codes in the community.

        TODO: Rewrite the implementation of this function.

        Args:
            img_shape (tuple[int]): Size of the image.
            num_vertices (int | tuple[int]): Min and max number of vertices. If
                only give an integer, we will fix the number of vertices.
                Default: (4, 8).
            max_angle (float): Max value of angle at each vertex. Default 4.0.
            length_range (int | tuple[int]): (min_length, max_length). If only give
                an integer, we will fix the length of brush. Default: (10, 100).
            brush_width (int | tuple[int]): (min_width, max_width). If only give
                an integer, we will fix the width of brush. Default: (10, 40).
            dtype (str): Indicate the data type of returned masks. Default: 'uint8'

        Returns:
            numpy.ndarray: Mask in the shape of (h, w, 1).
        """

        h, w = img_shape[:2]

        mask = np.zeros((h, w), dtype=dtype)
        if isinstance(length_range, int):
            min_length, max_length = length_range, length_range + 1
        elif isinstance(length_range, tuple):
            min_length, max_length = length_range
        else:
            raise TypeError('The type of length_range should be int'
                            f'or tuple[int], but got type: {length_range}')
        if isinstance(num_vertices, int):
            min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
        elif isinstance(num_vertices, tuple):
            min_num_vertices, max_num_vertices = num_vertices
        else:
            raise TypeError('The type of num_vertices should be int'
                            f'or tuple[int], but got type: {num_vertices}')

        if isinstance(brush_width, int):
            min_brush_width, max_brush_width = brush_width, brush_width + 1
        elif isinstance(brush_width, tuple):
            min_brush_width, max_brush_width = brush_width
        else:
            raise TypeError('The type of brush_width should be int'
                            f'or tuple[int], but got type: {brush_width}')

        num_v = np.random.randint(min_num_vertices, max_num_vertices)

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            # from the start point, randomly select n \in [1, 6] directions.
            direction_num = np.random.randint(16, 32)
            angle_list = np.random.randint(0, max_angle, size=direction_num)
            length_list = np.random.randint(
                min_length, max_length, size=direction_num)
            brush_width_list = np.random.randint(
                min_brush_width, max_brush_width, size=direction_num)
            for direct_n in range(direction_num):
                angle = 0.01 + angle_list[direct_n]
                if i % 2 == 0:
                    angle = 2 * math.pi - angle
                length = length_list[direct_n]
                brush_w = brush_width_list[direct_n]
                # compute end point according to the random angle
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1, brush_w)
                start_x, start_y = end_x, end_y
        mask = np.expand_dims(mask, axis=2)

        return mask

    def _retrieve_irregular_mask(self, img, area_ratio_range=(0.05, 0.5), **kwargs):
        """Get irregular mask with the constraints in mask ratio
        copied from: https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/136b29f58d0af6e5db9f3655d2891f5a855fcdaa/data/util/mask.py#L319

        Args:
            img (Tensor): Image [B, C, H, W].
            area_ratio_range (tuple(float)): Contain the minimum and maximum area
            ratio. Default: (0.15, 0.5).

        Returns:
            numpy.ndarray: Mask in the shape of (h, w, 1).
        """
        img_shape = (img.shape[2], img.shape[3])
        mask = self.random_irregular_mask(img_shape, **kwargs)
        min_ratio, max_ratio = area_ratio_range

        while not min_ratio < (np.sum(mask) /
                               (img_shape[0] * img_shape[1])) < max_ratio:
            mask = self.random_irregular_mask(img_shape, **kwargs)

        mask_b = torch.tensor(mask, device=img.device).permute(2, 0, 1).repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return (mask - 1) * -1

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask
        elif self.mask_type == 'free_form':
            mask = self._retrieve_irregular_mask(img)
            return mask


 
"""
For CT Reconstruction:
"""
def ct_parallel_project_2d(img, theta):
	bs, h, w, c = img.size()

	# (y, x)=(i, j): [0, w] -> [-0.5, 0.5]
	y, x = torch.meshgrid([torch.arange(h, dtype=torch.float32) / h - 0.5,
							torch.arange(w, dtype=torch.float32) / w - 0.5])

	# Rotation transform matrix: simulate parallel projection rays
	x_rot = x * torch.cos(theta) - y * torch.sin(theta)
	y_rot = x * torch.sin(theta) + y * torch.cos(theta)

	# Reverse back to index [0, w]
	x_rot = (x_rot + 0.5) * w
	y_rot = (y_rot + 0.5) * h

	# Resample (x, y) index of the pixel on the projection ray-theta
	sample_coords = torch.stack([y_rot, x_rot], dim=0).cuda()  # [2, h, w]
	img_resampled = map_coordinates(img, sample_coords) # [b, h, w, c]

	# Compute integral projections along rays
	proj = torch.mean(img_resampled, dim=1, keepdim=True) # [b, 1, w, c]

	return proj


def ct_parallel_project_2d_batch(img, thetas):
    '''
    img: input tensor [B, H, W, C]
    thetas: list of projection angles
    '''
    projs = []
    for theta in thetas:
      proj = ct_parallel_project_2d(img, theta)
      projs.append(proj)
    projs = torch.cat(projs, dim=1)  # [b, num, w, c]

    return projs