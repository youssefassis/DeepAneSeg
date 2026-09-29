import deepaneseg.volume.edition as ve
import numpy as np
import skimage.measure as skme
import skimage.morphology as skmo

def confusion_matrix(pred,vox2met,spheres, min_size=None, tp_proportion=0.3):
    '''
    takes a (binary) prediction volume with its vox2met transform and a set of ground truth spheres
    returns the TP,FN,FP, Cmat(voxels) for this prediction
    The prediction is labelled to get predicted connected components (PCC). 
    The ground truth is generated as a set of ground truth spheres (GTS).
    The intersections between each GTS and each PCC are computed.
    A TP is counted when an intersection is at least tp_proportion of either the GTS or the PCC volume
    A FN is counted when a GTS has no such intersection
    A FP is counted when a PCC has no such intersection
    Note that the case where 2 PCC intersection the same GTS has to be handled. For now we assume they both count for one TP
    Note that TN cannot be defined using the above. Thereafter, the function also returns Cmat, that is the confusion
    matrix computed on a voxel basis (e.g. TP==voxel is both positive in the GT and the prediction). Cmat=[[TP,FP],[FN,TN]]
    params:
    @pred: prediction volume (as returned by the model.predict, followed by np.where with a threshold)
    @vox2met: voxel 2 metric transform (see dio.read_nii_from_file)
    @spheres: ground truth spheres used to annotate aneurysms (see dio.points_to_spheres)
    @min_size: if not None, CC with a size smaller than min_size (in voxels) will be removed from prediction before computation
    @tp_proportion: minimum intersection to consider a TP (proportion of ground truth sphere)
    '''
    # label CC in prediction
    label_pred=skme.label(pred.astype(np.uint8))
    if not min_size is None: # also perform relabelling to keep consecutive labels
        label_pred=skme.label(skmo.remove_small_objects(label_pred,min_size=min_size))
    ncc_pred=np.max(label_pred.ravel())

    # label CC in truth volume
    t = ve.get_truth(pred,vox2met,spheres)
    ncc_truth = np.max(t.ravel())

    if ncc_pred==0: # no detection
        h_vox=np.bincount(t.ravel())
        return 0,ncc_truth,0,np.array([[0,0],[np.sum(h_vox[1:]),h_vox[0]]])

    # histograms
    hist_truth,_=np.histogram(t.ravel(),bins=ncc_truth+1)
    hist_pred,_=np.histogram(label_pred.ravel(),bins=ncc_pred+1)
    hist_cross,_,_=np.histogram2d(label_pred.ravel(),t.ravel(),bins=[ncc_pred+1,ncc_truth+1])

    # voxel based confusion matrix
    tp_vox=np.sum(hist_cross[1:,1:])
    tn_vox=hist_cross[0,0]
    fp_vox=np.sum(hist_cross[1:,0])
    fn_vox=np.sum(hist_cross[0,1:])
    Cmat=np.array([[tp_vox,fp_vox],[fn_vox,tn_vox]])
 
    # removes the background component
    hist_truth=hist_truth[1:]
    hist_pred=hist_pred[1:]
    hist_cross=hist_cross[1:,1:]

    # analysis: check if each CC in truth intersects with at most 1 CC in prediction
    if np.max(np.sum(np.where(hist_cross>0,1,0), axis=0))>1:
        print(f'Problem: 2 PCC intersect a single GTS')

    # compute statistics
    # boolean array: h(i,j) is True iff PCC#i and GTS#j intersect as a TP
    h=np.where(np.maximum(hist_cross/hist_truth,(hist_cross.T/hist_pred).T)>=tp_proportion, 1, 0)
    TP = np.sum(np.max(h,axis=0)) # each column with at least a 1 in it
    FN=ncc_truth-TP # TP+FN=number of GTS
    FP=ncc_pred-np.sum(np.max(h,axis=1)) # number of lines with only 0s in them
    
    return TP, FN, FP, Cmat


def adam_evaluation(pred,vox2met,spheres, pretraited=None, min_size=None):
    '''
    takes a (binary) prediction volume with its vox2met transform and a set of ground truth spheres
    returns the TP and FP count for this prediction.
    The computation is performed according to the ADAM challenge
    params:
    @pred: prediction volume (as returned by the model.predict, followed by np.where with a threshold)
    @vox2met: voxel 2 metric transform (see dio.read_nii_from_file)
    @spheres: ground truth spheres used to annotate aneurysms (see dio.points_to_spheres)
    @pretraited: Pretraited aneurysms to be excluded as ADAM criteria
    @min_size: if not None, CC with a size smaller than min_size (in voxels) will be removed from prediction before computation
    '''
    # label CC in prediction
    label_pred=skme.label(pred.astype(np.uint8))
    if not min_size is None: # also perform relabelling to keep consecutive labels
        label_pred=skme.label(skmo.remove_small_objects(label_pred,min_size=min_size))
    ncc_pred=np.max(label_pred.ravel())

    # Exculude pretraited aneurysms
    if not pretraited is None:
        print("pretraited")
        for i in range(1, ncc_pred+1):
            tmp = label_pred.copy()
            tmp[tmp != i] = 0
            if np.multiply(tmp, pretraited).sum() > 0:
                label_pred[label_pred==i]=0

    # pred coords
    pred_locations = np.array([np.mean(np.where(label_pred==i),axis=1) for i in range(1,ncc_pred+1)])
    if pred_locations.size == 0:
        pred_coords = pred_locations.reshape(0, 3)
    else:
        pred_coords = pred_locations@vox2met[:3,:3].T + vox2met[:3,3]

    # test coords
    test_radii = spheres[:, -1]
    test_coords = spheres[:,:3]
    if test_coords.size == 0:
        test_coords = test_coords.reshape(0, 3)
    print(f'Case: {len(test_radii)} aneurysms, {pred_coords.shape[0]} detections')
    #True positives lie within radius  of true aneurysm. Only count one true positive per aneurysm. 
    true_positives = 0
    tp_diam=[] # list for the diameters of the TP (detected) aneurysms
    fn_diam=[] # list for the diameters of the FN (missed) aneurysms
    for location, radius in zip(test_coords, test_radii):
        detected = False
        for detection in pred_coords:
            distance = np.linalg.norm(detection - location)
            if distance <= radius:
                detected = True
        if detected:
            true_positives += 1
            tp_diam.append(2*radius)
        else:
            fn_diam.append(2*radius)

    false_positives = 0
    for detection in pred_coords:
        found = False
        for location, radius in zip(test_coords, test_radii):
            distance = np.linalg.norm(location - detection)
            if distance <= radius:
                found = True
        if not found:
            false_positives += 1

    return true_positives, false_positives, tp_diam, fn_diam
