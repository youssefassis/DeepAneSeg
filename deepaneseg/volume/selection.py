import numpy as np
import sklearn.neighbors as skn
import scipy.ndimage as sndi
import skimage.measure as skme


def extract_points_thres(vol, vox2met, thres_low, thres_high):
    """
    return points in vol whose value is between thres_low (strictly)
    and thres_high (loosely)
    return points coordinates (p) and values at these points (v)
    p is returned as a Nx3 array (or D is len(vol.shape) == D) of voxel centers in metric coordinates
    (NIfTI convention: voxel index i is centered at vox2met @ [i, 1]), and v as a 1xN-array
    """
    idx = np.nonzero(np.logical_and(vol > thres_low, vol <= thres_high))
    v = vol[idx]
    p = vox2met[:3, :] @ np.vstack((idx, np.ones(len(v))))

    return p.T, v


def points_in_radius(q, p, r):
    """
    return the indices of all points from p within distance r from q
    (p=point set, q=query point)
    p should be a Nx3 array and q a 3-point or Nx3 array of points
    (work ok in dimensions D!=3 too)
    """
    if len(q.shape) == 1:
        queries = np.array(q)[np.newaxis, :]
    else:
        queries = q
    tree = skn.KDTree(p)
    return np.unique(np.concatenate(tree.query_radius(queries, r)))


def select_points(
    vol, vox2met, thres_low, r, thres_high=None, forbidden_points=None, nb_points=None, extract_type="Vessels"
):
    """
    return p: where p is a Nx3 array of the coordinates of points above threshold thres,
              such that any two pair of points are at least at a distance r
    If forbidden_points is not None, it should be a Nx3 array of forbidden points (coordinates).
    In that case, all points within a distance of each point in that list are
    removed.
    If nb_points is provided (a number), then at most nb_points are returned.
    The function starts with the brightest allowable point and review points with
    decreasing value. As a consequence, v should be sorted (decreasing) on output

    Example use:
    d=ni.load('volume.nii')
    vol=np.asarray(d.dataobj)
    vox2met=vol.affine
    T=np.percentile(vol,95) # threshold to get the 5% brigtest voxels
    forbidden_points=IO.readFcsv('markers.fcsv')
    p=select_points(vol,vox2met,T,20,forbidden_points.values(),100)
    -> this extracts the 100 brightest points, among the 5% brightest points in the
       volume stored in 'volume.nii' such that no two points are within a distance
       20 mm from each other and no point is within 20 mm from points read in file
       'markers.fcsv'
    """
    if thres_high is None:
        thres_high = np.max(vol.ravel())
    p, v = extract_points_thres(vol, vox2met, thres_low, thres_high)

    if (nb_points is None) or (nb_points > len(v)):
        nb_points = len(v)

    if extract_type == "Vessels":
        order = np.argsort(v)[::-1]  # pick points with decreasing voxel values
    else:
        order = np.arange(len(v))
        np.random.shuffle(order)  # pick points randomly
    removed = np.zeros(len(v), dtype=bool)
    tree = skn.KDTree(p)

    if not forbidden_points is None:
        if len(forbidden_points.shape) == 1:
            forbidden_points = forbidden_points[np.newaxis, :]
        for i in tree.query_radius(forbidden_points, r):
            removed[i] = True

    ret = np.empty((0, 3))

    n = 0
    for idx in order:
        if not removed[idx]:
            q = p[idx, :].copy()
            i = tree.query_radius(q[np.newaxis, :], r)[0]
            removed[i] = True
            ret = np.vstack((ret, q[np.newaxis, :]))
            n = n + 1
            if n == nb_points:
                return ret
    return ret


def get_ball(r):
    """
    return a structure element shaped as a ball of radius r.
    can be used with skimage.morphology operators
    """
    x, y, z = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return ((x * x + y * y + z * z) <= r * r).astype(np.uint8)


def fill_between_edges(edges):
    """
    For each line along the first axis, fills every voxel between its first and last edge voxels (inclusive).
    edges: boolean volume; returns a uint8 mask of the same shape.
    """
    has_edge = edges.any(axis=0)
    first = np.argmax(edges, axis=0)
    last = edges.shape[0] - 1 - np.argmax(edges[::-1], axis=0)
    x = np.arange(edges.shape[0]).reshape((-1,) + (1,) * (edges.ndim - 1))
    return ((x >= first) & (x <= last) & has_edge).astype(np.uint8)


def remove_skull_mask(vol, percent=60):
    """
    Brain mask: the volume between the outermost strong edges of each line (gradient magnitude above its
    percent-th percentile), eroded to remove the skull.
    """
    edges = sndi.gaussian_gradient_magnitude(vol, sigma=3)
    mask = fill_between_edges(edges >= np.percentile(edges, percent))

    # erode this mask to remove the skull
    footprint = get_ball(2)
    for _ in range(15):
        mask = sndi.binary_erosion(mask, structure=footprint, border_value=True).astype(np.uint8)
    # return this mask
    return mask


def skull_strip(vol):
    """Skull-stripped volume, rescaled by its maximum (as written to 'noskull volume' by remove_skull.py)."""
    vol = np.asarray(vol, dtype=np.float32)
    vol = remove_skull_mask(vol) * vol
    if np.max(vol) <= 0:
        raise ValueError(f"skull stripping left no brain voxel in a volume of shape {vol.shape}")
    return vol / np.max(vol)


def connected_components_to_spheres(vol, vox2met):
    """
    Takes a binary volume and returns a sphere for each connected components
    """
    # label CC in volume
    labels = skme.label(vol.astype(np.uint8))
    ncc_pred = np.max(labels.ravel())
    spheres = np.empty((0, 4))
    for i in range(ncc_pred):
        # extract positions of voxels in CC (in voxel coords)
        vpos = np.vstack(np.where(labels == i + 1))
        # transform in metric coords
        mpos = (vox2met @ np.vstack((vpos, np.ones(vpos.shape[1]))))[:3, :]
        # compute center of gravity
        cog = np.mean(mpos, axis=1)
        # compute the radius
        m = np.min(mpos, axis=1)
        M = np.max(mpos, axis=1)
        r = np.max(M - m) / 2

        # alternative: through volume
        #        voxel_size = np.linalg.norm(vox2met[:3,:3],axis=0)
        #        voxel_volume = np.prod(voxel_size)
        #        sphere_volume = vpos.shape[1] * voxel_volume
        #        r = (sphere_volume * 3/(4*np.pi))**(1/3)

        spheres = np.vstack((spheres, np.hstack((cog, r))))
    return spheres
