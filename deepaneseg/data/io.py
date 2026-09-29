import numpy as np
import os
import nibabel as ni
import string
import pandas as pd
import json, glob

from deepaneseg.volume.selection import select_points
from deepaneseg.utils import get_logger

logger = get_logger("Data Preparation")
# the chosen file format is python pickle dump of a dict. This might not be the wisest choice, but 
# hdf5 format is not reentrant, so that you cannot leverage threads to train model (seg fault)
# csv might be nice, but I am not convinced that tables are appropriate to store our data
# sql could be a very nice option... to consider when I have the time

# the data
# for each patient, we have:
# - an MRA: stored as nii.gz file -> extract the data matrix and the affine transform (4x4 matrix)
# - a list of segments (pairs of points): one for each aneurysm, the first point is the entry point 
#   and the second point is on the surface of the aneurysm so that the middle segment point is the 
#   center of the aneurysm and the segment length gives the diameter of the approximating sphere
# - a list of points, outside the aneurysms, where patches should extracted (location of the patch centers)
# each point is provided in a metric space, and the affine transform metric to voxel coordinate transform


def fetch_patient_dirs(base_dir):
    '''
    list all directories in base_dir that meet name format 'P????' and contain a file named config.json
    primary use it to generate a list of patient directory names, with base_dir=config['base_dir'] (where
    config is read from a base config.json file, see PreprocessData.ipynb)
    '''
    return [os.path.dirname(subject_conf) for subject_conf in glob.glob(os.path.join(base_dir, 'P????', 'config.json'))]


def read_nii_from_file(fname):
    '''
    Read a nii volume and return a 2-tuple with the voxel data (np array) and the voxel2metric tranform
    '''
    nv = ni.load(fname)
    return nv.get_fdata(), nv.affine

def save_nii_to_file(fname,vol,vox2met):
    ni.save(ni.Nifti1Image(vol, vox2met), fname)

def read_points_from_csv(fname):
    '''
    returns a list of points, read from a csv file. Coordinates are stored in columns named 'x', 'y' and 'z'
    '''
    try:
        d = pd.read_csv(fname, usecols=['x','y','z'])
        return d.to_numpy()
    except:
        return None

def points_to_spheres(p):
    '''
    p: list of 3D points as a Nx3 array, where N is even, so that every two points are considered as segment
    returns: a list of N/2 x 4 values: the first 3 values are the center of each segment (i.e. center of the sphere),
             and the 4th value is its half-length (i.e. radius of the sphere)
    '''
    if p.shape[1] != 3:
        raise ValueError('Input array must be a list of 3D points')
    if p.shape[0]%2 != 0:
        raise ValueError('Input array must contain an even number of points')

    c = 0.5*(p[::2]+p[1::2])
    r = np.linalg.norm(p[::2]-p[1::2],axis=1)/2
    return np.hstack((c,r.reshape((-1,1))))

def read_patient_data(vol_file, ane_file, normalize=None, vessel_file=None):
    '''
    returns a dict with keys 'data','affine','aneurysms'
    'data': volume voxel data, as returned by read_nii_from_file(vol_file)
    'affine': voxel2metric affine transform, as returned by read_nii_from_file(vol_file)
    'aneurysms': 2Nx3 array of aneurysms defining points (one entry and one end point for each aneurysm, hence the 2N)
    'normalize': volume data normalization. Accepted values are:
        - None or 'None'->no normalization
        - 'Linear': linear rescaling between 0 and 1
        - 'Normal': rescaling to 0-mean and unit-variance data
    '''
    d, a = read_nii_from_file(vol_file)
    if not vessel_file is None:
        seg, _ = read_nii_from_file(vessel_file)
        seg = np.where(seg>0,1,0).astype(np.uint8)   # just in case
    else:
        seg = None
    if not ane_file is None:
        s = read_points_from_csv(ane_file)
    else:
        s = None

    # should we use scikit.learn.normalize (or transform)?
    if normalize == 'Linear':
        m=np.min(d.ravel())
        M=np.max(d.ravel())
        d=(d-m)/(M-m)
    elif normalize == 'Normal':
        m=np.mean(d.ravel())
        std=np.std(d.ravel())
        d=(d-m)/std

    return {'data':d, 'affine':a, 'aneurysms':s, 'vessel':seg}

def read_patient_data_from_config(cfg_name, volume='init volume', normalize=None, vessel=None):
    '''
    same as read_patient_data, but vol_file and ane_file are read from a json config file (cfg_name) under keys
    'init volume' and 'pts aneurysm'
    '''
    dir=os.path.dirname(cfg_name)
    with open(cfg_name,'r') as f:
        c=json.load(f)
    if not vessel is None and os.path.isfile(os.path.join(dir, dir.split("/")[-1]+".nii.gz")): #c['vessel']))
        vessel_file = os.path.join(dir, dir.split("/")[-1]+".nii.gz") #c['vessel'])
    else:
        vessel_file = None
    return read_patient_data(os.path.join(dir,c[volume]),os.path.join(dir,c['pts aneurysm']) if 'pts aneurysm' in c else None, normalize=normalize, vessel_file=vessel_file)

def read_patient_data_base(pat_list, volume='init volume', normalize=None, vessel=None, label=None):
    '''
    return a list of dictionaries providing data about each patient located in a list a directories (e.g. generated by fetch_patient_dirs())
    each element is a dict with keys 'dir' (the patient directory on disk), 'affine', data, 'aneurysms' (see read_patient_data() for more info)
    '''
    logger.info(f"Loading {len(pat_list)} patients for '{label}' (vessel={volume}, '{normalize}' normalization, vessel is set to {vessel})")
    return [dict({'dir': p}, **(read_patient_data_from_config(os.path.join(p,'config.json'), volume=volume, normalize=normalize, vessel=vessel))) for p in pat_list]

def add_points_to_patient_data(da,pname):
    '''
    takes a patient data_base stored in an array of dictionaries (da) and adds a new key 'points' to each dictionary whose value is a
    Nx3 np.array of 3D points, as read in a file named pname in each patient directory (stored in key 'dir').
    Points should be stored in a CSV file with columns 'x','y' and 'z' (see read_points_from_csv()).
    The input array of dictionary is updated and returned.
    See read_patient_data_base() for more info on the dictionary structure.
    '''
    for d in da:
        d.update({'points':read_points_from_csv(os.path.join(d['dir'],pname))})
    return da

def save_split(cfg, train_list, valid_list, test_list):
    '''
    saves a list of patients (directories) split in train, validation and test lists into a json file (cfg), with keys 'training list', 
    'validation list' and 'testing list'
    If the file already exists, it is updated, eventually with new keys as above, else it is created with these three keys.
    '''
    try:
        with open(cfg,'r') as f:
            d = json.load(f)
    except:
        d = {}
    d['training list'] = train_list
    d['validation list'] = valid_list
    d['testing list'] = test_list
    with open(cfg, 'w') as f:
        json.dump(d, f, indent=2)

def read_split(cfg):
    '''
    reads a split list of patients (directories), in a json file with keys 'training list', 'validation list' and 'testing list'
    see also save_split()
    '''
    with open(cfg, 'r') as f:
        d = json.load(f)
    logger.info(f"Data splits: Training {len(d['training list'])}, Validation: {len(d['validation list'])}, Testing: {len(d['testing list'])}")
    return d['training list'], d['validation list'], d['testing list']


def csv2fcsv(csv_pathname):
    filename, file_extension = os.path.splitext(csv_pathname)
    assert file_extension == '.csv', "Please choose a file '.csv' extension"
    fcsv_file_out = filename + ".fcsv"
    pcsv = pd.read_csv(csv_pathname)
    label=[f'{t}_{i}' for i,t in pcsv[['Unnamed: 0','type']].to_numpy()]
    df = pd.concat([pcsv,pd.Series(label,name='id')], axis=1)
    df = df[['id','x','y','z']]
    head=f'# Markups fiducial file version = 4.10\n# CoordinateSystem = 0\n# columns = {",".join(df)}'
    with open(fcsv_file_out, 'w') as f:
	    f.write(head)
	    for i,l in df.iterrows():
	    	f.write(','.join([f'{v}' for v in l])+'\n')


def fcsv2csv(fcsv_file):
    filename, file_extension = os.path.splitext(csv_file)
    assert file_extension == '.fcsv', "Please choose a file with '.fcsv' extension"
    csv_file_out = filename + ".csv"

    with open(fcsv_file, 'r') as f:
        l=f.readlines()
    # coordinate system
    cs=l[1].strip().split('=')[1].strip()
    # column names
    cn=list(map(str.strip,l[2].split('=')[1].split(',')))
    # data
    data=[s.strip().split(',') for s in l[3:]]
    # check coordinate systems and column names
    # RAS: if cs='RAS' or cs='0'
    # LPS: if cs='LPS' or cs='1'
    # not checking for position column names (could be x,y,z or l,p,s or r,a,s
    if cs == 'LPS' or cs == '1':
        for d in data:
            d[1] = str(-float(d[1]))
            d[2] = str(-float(d[2]))
    # change position column names
    cn[1:4] = ['x','y','z']
    # create data frame
    d=pd.DataFrame(data=data, columns=cn)
    # save the result
    d.to_csv(csv_file_out, index=False)

def extract_points_from_patient(patient, r=20, nb_points=100, outfile="points.csv", random_points=False):
	os.chdir(patient)
	print('Load volume from disk: '+patient)
	with open("config.json", 'r') as f:
		d = json.load(f)
	vol, vox2met = read_nii_from_file(d['noskull volume'])
	print(f'Extracting points')
	forbidden_points = read_points_from_csv(d['pts aneurysm'])
	fp = points_to_spheres(forbidden_points)[:,:-1] # drop centers
	q = None
	if random_points:
		min_threshold, max_threshold = np.percentile(vol[vol>0], 0), np.percentile(vol[vol>0], 100)
		print(f'\Random Points ', end='')
		p=select_points(vol, vox2met, thres_low=min_threshold, thres_high=max_threshold, r=r,
                             forbidden_points=fp, nb_points=nb_points, extract_type='Parenchyma')
		print(f'{len(p)} random points')
		title = 'Random'
	else:
		T= np.percentile(vol[vol>0], 95)
		min_threshold = np.percentile(vol[vol>0], 50)
		max_threshold = np.percentile(vol[vol>0], 90)
		p = select_points(vol, vox2met, thres_low=T, r=r, forbidden_points=fp, nb_points=nb_points, extract_type='Vessels')
		title = 'Vessel'
		print(f'{len(p)} vessel points')
		q=select_points(vol,vox2met,thres_low=min_threshold,thres_high=max_threshold,r=r,forbidden_points=np.vstack((fp,p)),
                                nb_points=nb_points,extract_type='Parenchyma')
		print(f'{len(q)} tParenchyma points')

	# export to csv using pandas
	#print(f'Exporting to CSV')
	ps = pd.DataFrame(p, columns=list('xyz'))
	ps['type'] = pd.Categorical([title] * len(p))
	if q is not None:
		qs = pd.DataFrame(q, columns=list('xyz'))
		qs['type']=pd.Categorical(['Parenchyma']*len(q))
		points=ps.append(qs)
	else:
		points = ps
	points.to_csv(outfile)
	csv2fcsv(outfile)

def extract_points(data_path, r = 20, nb_points=100, outfile="points.csv", random_points = False):
    patients = fetch_patient_dirs(data_path)
    for patient in patients:
        extract_points_from_patient(patient, r=r, nb_points=nb_points, outfile=outfile, random_points=random_points)

#extract_points("/srv/storage/tangram@talc-data2.nancy.grid5000.fr/yassis/Data/", r=20, nb_points=200, outfile="RandomPoints.csv", random_points=True)

