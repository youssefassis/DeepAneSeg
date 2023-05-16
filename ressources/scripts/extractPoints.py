#!/usr/bin/env python3
import numpy as np
import json
import Volume.Selection as vs
import Data.IO as dio
import pandas as pd

r = 20
nbPoints = 100

patients = dio.fetch_patient_dirs("/home/yassis/Data")
for pateint in patients:
	os.chdir(patient)

	print('Load volume from disk')
	with open("ndl_config.json", 'r') as f:
	    d = json.load(f)
	vol, vox2met = dio.read_nii_from_file(d['noskull volume'])

	print(f'Extracting points')
	fPoints=dio.read_points_from_csv(d['pts aneurysm'])
	fp = dio.points_to_spheres(fPoints)[:,:-1] # drop centers
	T=np.percentile(vol[vol>0],95)
	Tl=np.percentile(vol[vol>0],50)
	Th=np.percentile(vol[vol>0],90)
	print(f'\tVessels ', end='')
	p=vs.selectPoints(vol,vox2met,thresLow=T,r=r,fPoints=fp,nbPoints=nbPoints,extractType='Vessels')
	print(f'{len(p)} points')
	print(f'\tParenchyma ',end='')
	q=vs.selectPoints(vol,vox2met,thresLow=Tl,thresHigh=Th,r=r,fPoints=np.vstack((fp,p)),nbPoints=nbPoints,extractType='Parenchyma')
	print(f'{len(q)} points')

	# export to csv using pandas
	print(f'Exporting to CSV')
	ps=pd.DataFrame(p,columns=list('xyz'))
	ps['type']=pd.Categorical(['Vessel']*len(p))
	qs=pd.DataFrame(q,columns=list('xyz'))
	qs['type']=pd.Categorical(['Parenchyma']*len(q))

	points=ps.append(qs)
	points.to_csv('points.csv')