import numpy as np

def draw_sphere(vol, vox2met, c, r, val=0):
    '''
     Draw ("burn") a 3D sphere in a volume:
         The sphere is given by its center c and radius r in RAS coordinates
         vox2met is the 4x4 affine transform between voxel and RAS coordinates
         val is the voxel value used for burnt voxels
    '''

    met2vox=np.linalg.inv(vox2met)
    # first find a bounding box to extract a voxel patch
    c=np.asarray(c)
    pminus=met2vox[:3,:]@np.append(c-r,1)
    pplus=met2vox[:3,:]@np.append(c+r,1)
    pmin=np.minimum(pminus,pplus)
    pmax=np.maximum(pminus,pplus)
    pmin=np.round(pmin).astype(np.int32)
    pmax=np.round(pmax).astype(np.int32)

    # bounding box for the volume
    vmin=np.zeros(3).astype(np.int32)
    vmax=np.asarray(vol.shape).astype(np.int32)-1

    # compute intersection
    pmin = np.maximum(pmin,vmin)
    pmax = np.minimum(pmax,vmax)+1 # previous points where included in intervals

    # check for non empty intersection
    if np.all(np.greater(pmax,pmin)):
        # compute distance patch
        # list of RAS coordinates for the points in in the bounding box
        x,y,z=np.mgrid[pmin[0]:pmax[0],pmin[1]:pmax[1],pmin[2]:pmax[2]]
        p=vox2met[:3,:]@np.vstack((x.ravel(),y.ravel(),z.ravel(),np.ones(len(x.ravel()))))
        # compute distance field
        d = np.linalg.norm(p-c[:,np.newaxis],axis=0)
        # compute indices for points in sphere
        i = np.where(d<=r)[0]
        # burn voxels
        vol[pmin[0]:pmax[0],pmin[1]:pmax[1],pmin[2]:pmax[2]].flat[i] = val

def draw_cube(vol, vox2met, c, s, val=0):
    '''
    Draws a cube of center c and dimension s in vol
    val is the value taken by voxels inside the cube
    '''
    met2vox=np.linalg.inv(vox2met)
    # first find a bounding box to extract a voxel patch
    c=np.asarray(c)
    pminus=met2vox[:3,:]@np.append(c-s/2,1)
    pplus=met2vox[:3,:]@np.append(c+s/2,1)
    pmin=np.minimum(pminus,pplus)
    pmax=np.maximum(pminus,pplus)
    pmin=np.round(pmin).astype(np.int32)
    pmax=np.round(pmax).astype(np.int32)
        # bounding box for the volume
    vmin=np.zeros(3).astype(np.int32)
    vmax=np.asarray(vol.shape).astype(np.int32)-1

    # compute intersection
    pmin = np.maximum(pmin,vmin)
    pmax = np.minimum(pmax,vmax)+1 # previous points where included in intervals
    vol[pmin[0]:pmax[0],pmin[1]:pmax[1],pmin[2]:pmax[2]] = val

def get_truth(vol,vox2met,spheres):
    '''
    Burns the spheres stored in spheres: each line contains 4 components
    The first 3 components are the center of the sphere and the 4 is its radius
    The resulting volume contains 0 as a background value, and the index in spheres
    (starting at 1) for the voxel in each sphere (labeled image). If two spheres
    intersect, the highest label is kept for the voxels in the intersection
    '''
    t=np.zeros(vol.shape).astype(np.uint8)
    if spheres is not None:
        for i,s in enumerate(spheres):
            draw_sphere(t,vox2met,s[:3],s[3],i+1)
    return t
