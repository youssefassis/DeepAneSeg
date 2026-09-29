#!/usr/bin/env python3
import os
import numpy as np
import json
import deepaneseg.volume.selection as vs
import deepaneseg.data.io as dio
import pandas as pd

r = 20
nb_points = 100

patients = dio.fetch_patient_dirs("/home/yassis/Data")
for patient in patients:
	os.chdir(patient)

	print('Load volume from disk')
	with open("ndl_config.json", 'r') as f:
	    d = json.load(f)
	vol, vox2met = dio.read_nii_from_file(d['noskull volume'])

	print(f'Extracting points')
	forbidden_points=dio.read_points_from_csv(d['pts aneurysm'])
	fp = dio.points_to_spheres(forbidden_points)[:,:-1] # drop radii
	T=np.percentile(vol[vol>0],95)
	Tl=np.percentile(vol[vol>0],50)
	Th=np.percentile(vol[vol>0],90)
	print(f'\tVessels ', end='')
	p=vs.select_points(vol,vox2met,thres_low=T,r=r,forbidden_points=fp,nb_points=nb_points,extract_type='Vessels')
	print(f'{len(p)} points')
	print(f'\tParenchyma ',end='')
	q=vs.select_points(vol,vox2met,thres_low=Tl,thres_high=Th,r=r,forbidden_points=np.vstack((fp,p)),nb_points=nb_points,extract_type='Parenchyma')
	print(f'{len(q)} points')

	# export to csv using pandas
	print(f'Exporting to CSV')
	ps=pd.DataFrame(p,columns=list('xyz'))
	ps['type']=pd.Categorical(['Vessel']*len(p))
	qs=pd.DataFrame(q,columns=list('xyz'))
	qs['type']=pd.Categorical(['Parenchyma']*len(q))

	points=pd.concat([ps, qs])
	points.to_csv('points.csv')