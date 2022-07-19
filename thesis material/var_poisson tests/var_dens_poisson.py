# from https://github.com/rmsouza01/iMRI-challenge/blob/master/Modules/sampling.py

import numpy as np
from numpy import linalg as LA


def centered_circle(image_shape, radius):
    """
    Description: creates a boolean centered circle image with a pre-defined radius
    :param image_shape: shape of the desired image
    :param radius: radius of the desired circle
    :return: circle image. It is a boolean image
    """

    center_x = int((image_shape[0] - 1) / 2)
    center_y = int((image_shape[1] - 1) / 2)

    X, Y = np.indices(image_shape)
    circle_image = ((X - center_x) ** 2 + (Y - center_y) ** 2) < radius ** 2  # type: bool

    return circle_image


def poisson_disc2d(pattern_shape, k, r):
    pattern_shape = (pattern_shape[0]-1, pattern_shape[1]-1)

    center = np.array([1.0 * pattern_shape[0] / 2,
                       1.0 * pattern_shape[1] / 2])
    width, height = pattern_shape

    # Cell side length (equal to r_min)
    a = 2

    # Number of cells in the x- and y-directions of the grid
    nx, ny = int(width / a), int(height / a)
    # A list of coordinates in the grid of cells
    coords_list = [(ix, iy) for ix in range(nx+1) for iy in range(ny+1)]
    # Initilalize the dictionary of cells: each key is a cell's coordinates, the
    # corresponding value is the index of that cell's point's that might cause
    # conflict when adding a new point.
    cells = {coords: [] for coords in coords_list}
    centernorm = LA.norm(center)


    def calc_r(coords):
        """Calculate r for the given coordinates."""
        return ((LA.norm(np.asarray(coords)-center) / centernorm) * 240 + 50) / r



    def get_cell_coords(pt):
        """Get the coordinates of the cell that pt = (x,y) falls in."""
        return int(pt[0] // a), int(pt[1] // a)

    def mark_neighbours(idx):
        """
        Add sample index to the cells within r(point) range of the point.
        """
        coords = samples[idx]
        if idx in cells[get_cell_coords(coords)]:
            # This point is already marked on the grid, so we can skip
            return
        rx = calc_r(coords)
        xvals = np.arange(coords[0] - rx, coords[0] + rx)
        yvals = np.arange(coords[1] - rx, coords[1] + rx)

        xvals = xvals[(xvals >= 0) & (xvals <= width)]
        yvals = yvals[(yvals >= 0) & (yvals <= height)]

        dist = lambda x, y: np.sqrt((coords[0] - x) ** 2 + (coords[1] - y) ** 2) < rx
        xx, yy = np.meshgrid(xvals, yvals, sparse=False)
        pts = np.vstack((xx.ravel(), yy.ravel())).T
        pts = pts[dist(pts[:, 0], pts[:, 1])]
        [cells[get_cell_coords(pt)].append(idx) for pt in pts]

    def point_valid(pt):
        """
		Is pt a valid point to emit as a sample?
        It must be no closer than r from any other point:
        check the points
		"""
        rx = calc_r(pt)
        if rx < 0.5:
            rx = 0.5
        neighbour_idxs = cells[get_cell_coords(pt)]
        for n in neighbour_idxs:
            n_coords = samples[n]
            #rn = calc_r(n_coords)
            # Squared distance between or candidate point, pt, and this nearby_pt.
            distance = np.sqrt((n_coords[0] - pt[0]) ** 2 + (n_coords[1] - pt[1]) ** 2)
            if distance < rx:
                # The points are too close, so pt is not a candidate.
                return False
        # All points tested: if we're here, pt is valid
        return True

    def get_point(k, refpt):
        """
        	Try to find a candidate point relative to refpt to emit in the sample.
		We draw up to k points from the annulus of inner radius r, outer
		radius 2r around the reference point, refpt. If none of
		them are suitable (because they're too close to existing points in
		the sample), return False. Otherwise, return the pt.
		"""
        i = 0
        rx = calc_r(refpt)
        while i < k:
            rho, theta = np.random.uniform(rx, 2 * rx), np.random.uniform(0, 2 * np.pi)
            pt = refpt[0] + rho * np.cos(theta), refpt[1] + rho * np.sin(theta)
            if not (0 < pt[0] < width and 0 < pt[1] < height):
                # Off the grid, try again.
                continue
            if point_valid(pt):
                return pt
            i += 1
        # We failed to find a suitable point in the vicinity of refpt.
        return False

    # Pick a random point to start with.
    pt = (np.random.uniform(0, width), np.random.uniform(0, height))
    samples = [pt]
    cursample = 0
    mark_neighbours(0)
    # Set active, in the sense that we're going to look for more points
    # in its neighbourhood.
    active = [0]
    # As long as there are points in the active list, keep trying to find samples.
    while active:
        # choose a random "reference" point from the active list.
        idx = np.random.choice(active)
        refpt = samples[idx]
        # Try to pick a new point relative to the reference point.
        pt = get_point(k, refpt)
        if pt:
            # Point pt is valid: add it to the samples list and mark it as active
            samples.append(pt)
            cursample += 1
            active.append(cursample)
            mark_neighbours(cursample)
        else:
            # We had to give up looking for valid points near refpt, so remove it
            # from the list of "active" points.
            active.remove(idx)
    samples = np.rint(np.array(samples)).astype(int)
    samples = np.unique(samples[:, 0] + 1j * samples[:, 1])
    samples = np.column_stack((samples.real, samples.imag)).astype(int)
    poisson_pattern = np.zeros((pattern_shape[0] + 1, \
                                pattern_shape[1] + 1), dtype=bool)
    poisson_pattern[samples[:, 0], samples[:, 1]] = True
    return poisson_pattern


def poisson_disc_pattern(pattern_shape, center=True, radius=10, k=10, r=50):
    """
    Description: creates a uniformly distributed sampling pattern.
    :param pattern_shape: shape of the desired sampling pattern.
    :param center: boolean variable telling whether or not sample low frequencies
    :param radius: variable telling radius (2D) to be sampled in the centre.
    :param k: Number of points around each reference point as candidates for a new
    sample point
    :param r: Minimum distance between samples.
    :return: sampling pattern. It is a boolean image
    """

    if center == False:
        return poisson_disc2d(pattern_shape, k, r)
    else:
        pattern1 = poisson_disc2d(pattern_shape, k, r)
        pattern2 = centered_circle(pattern_shape, radius)
        return np.logical_or(pattern1, pattern2)
