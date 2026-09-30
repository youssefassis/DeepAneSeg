import deepaneseg.data.io as dio
import deepaneseg.volume.selection as vs

import numpy as np
import os
import deepaneseg.inference.evaluation as DLe
import json

base_dir = "/path/to/ADAM/"
work_dir = os.path.join("/path/to/Data", "0Work")
train_dir = os.path.join(work_dir, "ndl_Train042")
with open(os.path.join(train_dir, "ndl_config.json"), "r") as f:
    config = json.load(f)

ddir = "/path/to/ADAM/"
test_pdir = os.path.join("/path/to/ndl_Train042", "predictions", "ADAM")  # ,'test') prediction/prediction_35/test/
# train_pdir=os.path.join(train_dir,'prediction','Train')
# valid_pdir=os.path.join(train_dir,'prediction','Valid')
# utilisation d'un map pour ne garder que le P???? et récupérer le nom du patient à partir du fichier P????.nii.gz
pat_names = sorted(
    list(map(lambda x: x[:5], os.listdir(test_pdir)))
)  # +list(map(lambda x:x[:5],os.listdir(valid_pdir)))+list(map(lambda x:x[:5],os.listdir(train_pdir))))
# pat_names = [ "P0133","P0134","P0135","P0136","P0137","P0138","P0139","P0140","P0141","P0142","P0143","P0144","P0145","P0146","P0147", "P0148", "P0149","P0150","P0151", "P0152"]

pat_names = [
    "P0153",
    "P0154",
    "P0155",
    "P0156",
    "P0157",
    "P0158",
    "P0159",
    "P0160",
    "P0161",
    "P0162",
    "P0163",
    "P0164",
    "P0165",
    "P0166",
    "P0167",
    "P0168",
    "P0169",
    "P0170",
    "P0171",
    "P0172",
    "P0173",
    "P0174",
    "P0175",
    "P0176",
    "P0177",
    "P0178",
    "P0179",
    "P0180",
    "P0181",
    "P0182",
    "P0183",
    "P0184",
    "P0185",
    "P0186",
    "P0187",
    "P0188",
    "P0189",
    "P0190",
    "P0191",
    "P0192",
    "P0193",
    "P0194",
    "P0195",
    "P0196",
    "P0197",
    "P0198",
    "P0199",
    "P0200",
    "P0201",
    "P0202",
    "P0203",
    "P0204",
    "P0205",
    "P0206",
    "P0207",
    "P0208",
    "P0209",
    "P0210",
    "P0211",
    "P0212",
    "P0213",
    "P0214",
    "P0215",
    "P0216",
    "P0217",
    "P0218",
    "P0219",
    "P0220",
    "P0221",
    "P0222",
    "P0223",
    "P0224",
    "P0225",
    "P0226",
    "P0227",
    "P0228",
    "P0229",
    "P0230",
    "P0231",
    "P0232",
    "P0233",
    "P0234",
    "P0235",
    "P0236",
    "P0237",
    "P0238",
    "P0239",
    "P0240",
    "P0241",
    "P0242",
    "P0243",
    "P0244",
    "P0245",
]
# pat_names = ['P0192']
# print(pat_names)

threshold = 1
tp_proportion = 0.3  # proportion d'intersection entre 2 CC pour les considérer comme positives
vmin = None  # pas de filtrage des CC selon leur volume

"""
# computation of perf both aneurysm- and voxel-wise
print("Voxel-wise metrics")
tp_tot = 0
fn_tot = 0
fp_tot = 0
cmat_tot = np.zeros((2,2))
for p in pat_names:
    print(f'{p}:',end='')
    fname = os.path.join(test_pdir, p+'.nii.gz')
#    if not os.path.exists(fname):
#        fname=os.path.join(valid_pdir,p+'.nii.gz')
    pred, vox2met = dio.read_nii_from_file(fname)
#    print(os.path.join(base_dir,p,'F.csv'))
    points = dio.read_points_from_csv(os.path.join(base_dir,p,'F.csv'))
    print(os.path.join(base_dir,p,'F.csv'), "vs", fname)
    spheres=dio.points_to_spheres(points) #dio.read_points_from_csv(os.path.join(base_dir,p,'F.csv')))
#    pred[pred<threshold]=0
#    pred[pred>=threshold]=1
    TP,FN,FP,Cmat=DLe.confusion_matrix(np.where(pred<threshold,0,1),vox2met,spheres,vmin,tp_proportion)
#    TP,FN,FP,Cmat=DLe.confusion_matrix(pred, vox2met, spheres, vmin, tp_proportion)
    print(f'{TP}, {FP}, {FN}, {Cmat}')
    tp_tot += TP
    fn_tot += FN
    fp_tot += FP
    cmat_tot += Cmat/1e5


# affichage des résultats
print(f'{tp_tot}, {fn_tot}, {fp_tot}, {cmat_tot}')
#sensitivity
print(f'Sensitivity: {tp_tot/(tp_tot+fn_tot)}')
#sens and spec, voxelwise
print(f'Sensitivity: {cmat_tot[0,0]/(cmat_tot[0,0]+cmat_tot[1,0])}, Specificity: {cmat_tot[1,1]/(cmat_tot[1,1]+cmat_tot[0,1])}')

#kappa (voxel)
tpv,fpv,fnv,tnv=cmat_tot.ravel()
glob=np.sum(cmat_tot)
Pacc=(tpv+tnv)/glob
Ppos=(tpv+fpv)*(tpv+fnv)/(glob*glob)
Pneg=(tnv+fnv)*(tnv+fpv)/(glob*glob)
Phas=Ppos+Pneg
Kappa=(Pacc-Phas)/(1-Phas)
print(f"Kappa_voxels: {Kappa}")


# Computation of perf patch-wise (à la Youssef)
# extract the patches around the points (both positive and negative) and count as positive any patch with positive voxels
print("Patch-wise metrics")
tp_tot=0
tn_tot=0
fn_tot=0
fp_tot=0
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
        v,t,_=vp.get_patch_and_truth(pred,vox2met,pt,size,dim,annot_pts)
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

    tp_tot += TP
    fp_tot += FP
    fn_tot += FN
    tn_tot += TN

#sensitivity and specificity
print(f'Sensitivity: {tp_tot/(tp_tot+fn_tot)}, Specificity: {tn_tot/(tn_tot+fp_tot)}')

#kappa (patch
tpv,fpv,fnv,tnv=tp_tot,fp_tot,fn_tot,tn_tot
glob=np.sum(cmat_tot)
Pacc=(tpv+tnv)/glob
Ppos=(tpv+fpv)*(tpv+fnv)/(glob*glob)
Pneg=(tnv+fnv)*(tnv+fpv)/(glob*glob)
Phas=Ppos+Pneg
Kappa=(Pacc-Phas)/(1-Phas)
print(f"Kappa: {Kappa}")
"""
print("ADAM Metrics")
# ADAM Challenge computation
sens_tot = 0
fp_tot = 0
tp_tot = 0
nb_ane = 0
for p in pat_names:
    print(f"{p}:", end="")
    fname = os.path.join(test_pdir, p + ".nii.gz")
    pred, vox2met = dio.read_nii_from_file(fname)

    aneurysm, _ = dio.read_nii_from_file(os.path.join(ddir, p, "aneurysms.nii.gz"))

    pretraited_aneurysm = aneurysm.copy()
    pretraited_aneurysm[pretraited_aneurysm < 2] = 0
    pretraited_aneurysm[pretraited_aneurysm == 2] = 1

    aneurysm[aneurysm > 1] = 0
    spheres = vs.connected_components_to_spheres(aneurysm, vox2met)
    #    spheres = dio.points_to_spheres(dio.read_points_from_csv(os.path.join(base_dir,p,'F.csv')))
    pretraited_aneurysm = pretraited_aneurysm if pretraited_aneurysm.sum() > 0 else None
    if pretraited_aneurysm is not None:
        while pred.shape != pretraited_aneurysm.shape:
            if pred.shape[0] != pretraited_aneurysm.shape[0]:
                pred = pred[:-1, :, :]
            if pred.shape[1] != pretraited_aneurysm.shape[1]:
                pred = pred[:, :-1, :]
            if pred.shape[2] != pretraited_aneurysm.shape[2]:
                pred = pred[:, :, :-1]
        print(pred.shape, pretraited_aneurysm.shape)

    TP, FP, _, _ = DLe.adam_evaluation(np.where(pred < threshold, 0, 1), vox2met, spheres, pretraited_aneurysm, vmin)
    sens = TP / spheres.shape[0]
    nb_ane += spheres.shape[0]
    print(f"{TP}, {FP}, {spheres.shape[0] - TP}")  # spheres.shape[0]-TP = TN
    sens_tot += sens
    fp_tot += FP
    tp_tot += TP
nb_pat = len(pat_names)
R = tp_tot / nb_ane
P = tp_tot / (tp_tot + fp_tot)
print(
    f"sensitivity={sens_tot / nb_pat}, FP count={fp_tot / nb_pat}, TP count={tp_tot / nb_ane}, R={R}, P={P}, F2={(5 * P * R) / (4 * P + R)}"
)
