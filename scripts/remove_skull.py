#!/usr/bin/env python3
import os
import sys
import numpy as np
import nibabel as ni
import json
import deepaneseg.data.io as dio
import deepaneseg.volume.selection as vs

patients = dio.fetch_patient_dirs("/home/yassis/Data")
for patient in patients:
	os.chdir(patient)
	print('Load volume from disk')
	with open('config.json','r') as f:
	    d=json.load(f)
	ni_vol=ni.load(d['init volume'])
	vol=np.asarray(ni_vol.dataobj).astype(np.float32)
	vox2met=ni_vol.affine
	met2vox=np.linalg.inv(vox2met)

	print('Preprocess volume')
	mask=vs.remove_skull_mask(vol)
	vol=mask*vol
	max=np.max(vol.ravel())
	vol/=max

	print('Saving output')
	outname='noskull.nii.gz'
	d['noskull volume']=outname
	ni.Nifti1Image(vol,vox2met).to_filename(outname)
	with open('config.json','w') as f:
	    json.dump(d,f,indent=2)