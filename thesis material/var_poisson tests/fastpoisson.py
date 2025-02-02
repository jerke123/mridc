import numpy as np
#import numba as nb

def poisson(img_shape, accel, calib=(0, 0), dtype=np.complex,
            crop_corner=True, return_density=False, seed=0,
            max_attempts=30, tol=0.1):
    """Generate variable-density Poisson-disc sampling pattern.
    The function generates a variable density Poisson-disc sampling
    mask with density proportional to :math:`1 / (1 + s |r|)`,
    where :math:`r` represents the k-space radius, and :math:`s`
    represents the slope. A binary search is performed on the slope :math:`s`
    such that the resulting acceleration factor is close to the
    prescribed acceleration factor `accel`. The parameter `tol`
    determines how much they can deviate.
    Args:
        img_shape (tuple of ints): length-2 image shape.
        accel (float): Target acceleration factor. Must be greater than 1.
        calib (tuple of ints): length-2 calibration shape.
        dtype (Dtype): data type.
        crop_corner (bool): Toggle whether to crop sampling corners.
        seed (int): Random seed.
        max_attempts (float): maximum number of samples to reject in Poisson
           disc calculation.
        tol (float): Tolerance for how much the resulting acceleration can
            deviate form `accel`.
    Returns:
        array: Poisson-disc sampling mask.
    References:
        Bridson, Robert. "Fast Poisson disk sampling in arbitrary dimensions."
        SIGGRAPH sketches. 2007.
    """
    if accel <= 1:
        raise ValueError(f'accel must be greater than 1, got {accel}')

    if seed is not None:
        rand_state = np.random.get_state()

    ny, nx = img_shape
    y, x = np.mgrid[:ny, :nx]
    x = np.maximum(abs(x - img_shape[-1] / 2) - calib[-1] / 2, 0)
    x /= x.max()
    y = np.maximum(abs(y - img_shape[-2] / 2) - calib[-2] / 2, 0)
    y /= y.max()
    r = np.sqrt(x**2 + y**2)

    slope_max = max(nx, ny)
    slope_min = 0
    while slope_min < slope_max:
        slope = (slope_max + slope_min) / 2
        radius_x = np.clip((1 + r * slope) * nx / max(nx, ny), 1, None)
        radius_y = np.clip((1 + r * slope) * ny / max(nx, ny), 1, None)
        mask = _poisson(
            img_shape[-1], img_shape[-2], max_attempts,
            radius_x, radius_y, calib, seed)
        if crop_corner:
            mask *= r < 1

        actual_accel = img_shape[-1] * img_shape[-2] / np.sum(mask)

        if abs(actual_accel - accel) < tol:
            break
        if actual_accel < accel:
            slope_min = slope
        else:
            slope_max = slope

    if abs(actual_accel - accel) >= tol:
        raise ValueError(f'Cannot generate mask to satisfy accel={accel}.')

    mask = mask.reshape(img_shape).astype(dtype)

    if seed is not None:
        np.random.set_state(rand_state)

    return mask
