import Data.IO as dio
#import Volume.Selection as vs
import Volume.Edition as ve
import Volume.Patch as vp
import skimage.morphology as skimo
import skimage.measure as skime
import numpy as np
import os
import DL.Evaluation as DLe
import json

base_dir='/home/yassis/Data'
work_dir=os.path.join(base_dir, '0Work')
train_dir=os.path.join(work_dir,'ndl_Train002')
with open(os.path.join(train_dir,'ndl_config.json'),'r') as f:
    config=json.load(f)

test_pdir=os.path.join(train_dir,'prediction','test')
valid_pdir=os.path.join(train_dir,'prediction','valid')
train_pdir = os.path.join(train_dir,'prediction', 'train')
# utilisation d'un map pour ne garder que le P???? et récupérer le nom du patient à partir du fichier P????.nii.gz
#pat_names=sorted(list(map(lambda x:x[:5],os.listdir(test_pdir)))+list(map(lambda x:x[:5],os.listdir(valid_pdir)))+list(map(lambda x:x[:5],os.listdir(train_pdir))))
pat_names=sorted(list(map(lambda x:x[:5],os.listdir(test_pdir))))

threshold=0.5
TP_proportion=0.1 # proportion d'intersection entre 2 CC pour les considérer comme positives
vmin=None # pas de filtrage des CC selon leur volume

# computation of perf both aneurysm- and voxel-wise
TP_tot=0
FN_tot=0
FP_tot=0
Cmat_tot=np.zeros((2,2))
for p in pat_names:
    print(f'{p}:',end='')
    fname=os.path.join(test_pdir,p+'.nii.gz')
    if not os.path.exists(fname):
        fname = os.path.join(valid_pdir,p+'.nii.gz')
    pred, vox2met = dio.read_nii_from_file(fname)
    spheres = dio.points_to_spheres(dio.read_points_from_csv(os.path.join(base_dir,p,'F.csv')))

    TP,FN,FP,Cmat=DLe.confusion_matrix(np.where(pred<threshold,0,1),vox2met,spheres,vmin,TP_proportion)
    print(f'{TP}, {FP}, {FN}, {Cmat}')
    TP_tot += TP
    FN_tot += FN
    FP_tot += FP
    Cmat_tot += Cmat/1e5

# affichage des résultats
print(f'{TP_tot}, {FN_tot}, {FP_tot}, {Cmat_tot}')
#sensitivity
print(f'{TP_tot/(TP_tot+FN_tot)}')
#sens and spec, voxelwise
print(f'{Cmat_tot[0,0]/(Cmat_tot[0,0]+Cmat_tot[1,0])}, {Cmat_tot[1,1]/(Cmat_tot[1,1]+Cmat_tot[0,1])}')

#kappa (voxel)
tpv,fpv,fnv,tnv=Cmat_tot.ravel()
glob=np.sum(Cmat_tot)
Pacc=(tpv+tnv)/glob
Ppos=(tpv+fpv)*(tpv+fnv)/(glob*glob)
Pneg=(tnv+fnv)*(tnv+fpv)/(glob*glob)
Phas=Ppos+Pneg
Kappa=(Pacc-Phas)/(1-Phas)
print(Kappa)

# Computation of perf patch-wise (à la Youssef)
# extract the patches around the points (both positive and negative) and count as positive any patch with positive voxels
TP_tot=0
TN_tot=0
FN_tot=0
FP_tot=0
size=config['patch_size']
dim=config['patch_shape']
for p in pat_names:
    print(f'{p}: ', end='')
    # prediction + threshold + cleaning if set
    fname=os.path.join(test_pdir,p+'.nii.gz')
    if not os.path.exists(fname):
        fname=os.path.join(valid_pdir,p+'.nii.gz')
    pred,vox2met=dio.read_nii_from_file(fname)
    pred=np.where(pred<threshold, 0, 1).astype(np.uint8)
    
    if not vmin is None:
        pred=skimo.remove_small_objects(skime.label(pred),min_size=vmin)
        pred=np.where(pred>0, 1, 0)
        
    # points
    pat_dir=os.path.join(base_dir,p)
    with open (os.path.join(pat_dir,'config.json'),'r') as f:
        pconfig=json.load(f)
    annot_pts=dio.read_points_from_csv(os.path.join(pat_dir,pconfig['pts aneurysm']))
    # aneurysm
    ane_pts=dio.points_to_spheres(annot_pts)[:,:3]
    # negative patch centers
    neg_pts=dio.read_points_from_csv(os.path.join(pat_dir,config['negative patch centers']))
    
    all_pts = np.vstack((ane_pts,neg_pts))

    TP,FP,FN,TN=0,0,0,0
    for pt in all_pts:
        v,t,_=vp.getPatchAndTruth(pred,vox2met,pt,size,dim,annot_pts)
        vmax=np.max(v.ravel())
        tmax=np.max(t.ravel())
        if tmax > 0:
            if vmax > 0: # TP
                TP += 1
            else: # FN
                FN += 1
        else:
            if vmax > 0: # FP
                FP += 1
            else: #TN
                TN += 1
    print(f'{TP}, {FP}, {FN}, {TN}')
    
    TP_tot += TP
    FP_tot += FP
    FN_tot += FN
    TN_tot += TN

#sensitivity and specificity
print(f'{TP_tot/(TP_tot+FN_tot)}, {TN_tot/(TN_tot+FP_tot)}')

#kappa (patch
tpv,fpv,fnv,tnv=TP_tot,FP_tot,FN_tot,TN_tot
glob=np.sum(Cmat_tot)
Pacc=(tpv+tnv)/glob
Ppos=(tpv+fpv)*(tpv+fnv)/(glob*glob)
Pneg=(tnv+fnv)*(tnv+fpv)/(glob*glob)
Phas=Ppos+Pneg
Kappa=(Pacc-Phas)/(1-Phas)
print(Kappa)

# ADAM Challenge computation
sens_tot=0
FP_tot=0
TP_tot=0
nb_ane=0
for p in pat_names:
    print(f'{p}:',end='')
    fname=os.path.join(test_pdir,p+'.nii.gz')
    if not os.path.exists(fname):
        fname=os.path.join(valid_pdir,p+'.nii.gz')
    if not os.path.exists(fname):
        fname=os.path.join(train_pdir,p+'.nii.gz')    
    pred,vox2met=dio.read_nii_from_file(fname)
    spheres=dio.points_to_spheres(dio.read_points_from_csv(os.path.join(base_dir,p,'F.csv')))

    TP,FP=DLe.ADAM_evaluation(np.where(pred<threshold,0,1),vox2met,spheres,vmin)
    sens=TP/spheres.shape[0]
    nb_ane+=spheres.shape[0]
    print(f'{TP}, {FP}, {spheres.shape[0]-TP}') # spheres.shape[0]-TP == TN
    sens_tot += sens
    FP_tot += FP
    TP_tot += TP
nb_pat=len(pat_names)
R=TP_tot/nb_ane
P=TP_tot/(TP_tot+FP_tot)
print(f'sensitivity={sens_tot/nb_pat}, FP count={FP_tot/nb_pat}, TP count={TP_tot/nb_ane}, R={R}, P={P}, F2={(5*P*R)/(4*P+R)}')

