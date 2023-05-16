#!/usr/bin/env python3
import os
import sys
import json
import torch

import Data.IO as dio
import Volume.Patch as vp

from utils import load_model, get_logger

from skimage import morphology, measure
import numpy as np

def main():
    d = sys.argv[1]
    cfg = os.path.join(d,'ndl_config.json')
    try:
        with open(cfg,'r') as f:
            config = json.load(f)
    except:
        print(f'No such config file {cfg}')
        exit()

    logger = get_logger('Evaluation')

   # get testing generator
    _, _, test_list = dio.readSplit(config['split_file'])
    normalize = config['normalize'] if 'normalize' in config else None
    test_db = dio.add_points_to_patient_data(dio.read_patient_data_base(test_list,normalize=normalize),
                                            config['negative patch centers'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config, logger)
    model.to(device)

    smallobj = 40
    threshold = 0.5
    prob = 0.3
    seuil = 50
#    for threshold in [0, 0.5, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    get_test_generator(test_db, config["patch_size"], config["patch_shape"], model, threshold = threshold, smallobj=smallobj, holes=0, prob=prob, seuil=seuil)

def getLargestNCC(image, N=1):
    """ image: binary ndarray"""
    labels = measure.label(image)
    max = np.bincount(labels.flat)[1:]
    CC = np.zeros(image.shape)
    for label in (-max).argsort()[:N]+1:
      CC+=labels == label
    return CC

def getNumberCC(image):
    return np.amax(measure.label(image))

def postProcessing(volume, threshold, rmsmallObjects= 0, holes=0):
    volume[volume <= threshold] = 0
    volume[volume > threshold] = 1

    # morphology operations
    volume = volume.astype('bool')
    volume = morphology.binary_opening(volume, morphology.square(4))
    volume = morphology.binary_closing(volume, morphology.square(4))
#    volume = morphology.remove_small_holes(volume, holes)
    if (rmsmallObjects != 0):
        volume = morphology.remove_small_objects(volume, min_size=rmsmallObjects)

    # Get largest connected components
    volume = getLargestNCC(volume, 1)
    return tf.cast(volume, dtype=tf.float32)

def predict(model, volume):
    dim = volume.shape
#    volume = tf.constant([[volume]])
    return model(torch.from_numpy(volume)).reshape(dim)

eps = 0.000001 #tf.keras.backend.epsilon()

def getSubPatches(image):
    labels = measure.label(image)
    arr = []
    for i in range(1, getNumberCC(image)+1):
        arr.append(np.asarray([labels==i], dtype=np.int) )
    return arr

def confusion_matrixPatches(v,t, v2m, prob=0.4):
    v = v.numpy()
    t = t.numpy()
    TP, FP, TN, FN = 0,0,0,0
    intersection = 0
    if t.sum() > 0: # GT as aneurysm
        if v.sum() == 0:
            FN = 1
        else:
            intersection = t[v==1].sum() / t.sum()
            if intersection >= prob:
                TP=1
            else:
                FN=1
    else: # GT as non aneurysm
        if v.sum() == 0:
            TN = 1
        else:
            FP = 1
    return {'tp': TP, 'tn': TN, 'fp': FP,'fn':FN}

def get_test_generator(pat_db, size, dim, model, threshold, smallobj=0, holes=0, prob=0.3, seuil=30):
    for patient in pat_db:
        print(patient['dir'])
        # Patches without aneurysms
        p_list = [{'point': p,'data': patient['data'],'vox2met':patient['affine'],'aneurysms':patient['aneurysms'],'status':False} for p in patient['points']]
        # Patches of aneurysms
        a_list = [{'point': p,'data': patient['data'],'vox2met':patient['affine'],'aneurysms':patient['aneurysms'],'status':True} for p in dio.points_to_spheres(patient['aneurysms'])]
        g_list = p_list + a_list

        TP = list()
        TN = list()
        FP = list()
        FN = list()

        # For each patch in 'patient'
        for p in g_list:
            v, t, v2m = vp.getPatchAndTruth(p['data'], p['vox2met'], p['point'], size, dim, p['aneurysms'], None, None)
            gt = [t]
            if getNumberCC(t) > 1 :
                gt = getSubPatches(t)
            v = predict(model, v)
            v = postProcessing(v, threshold, rmsmallObjects=smallobj, holes=holes)
            for aneurysmTruth  in gt:
                #tf.cast(aneurysmTruth, tf.float32)

                conf_mat = confusion_matrixPatches(v, aneurysmTruth, v2m, patient['dir'].split("/")[-1], prob=prob)
                TP.append(conf_mat['tp'])
                TN.append(conf_mat['tn'])
                FP.append(conf_mat['fp'])
                FN.append(conf_mat['fn'])
        tp = sum(TP)
        tn = sum(TN)
        fp = sum(FP)
        fn = sum(FN)
        # Kappa
        pa = (tn+tp)/(tn+fp+fn+tp) 
        P_a = ((tn+fp)/(tn+fp+fn+tp))*((tn+fn)/(tn+fp+fn+tp)) # {aneurysm vs aneurysm}
        P_n = ((fn+tp)/(tn+fp+fn+tp))*((fp+tp)/(tn+fp+fn+tp)) # {non-aneurysm vs non-aneurysm}
        pe = P_a + P_n
        kappascore = (pa-pe)/(1-pe)

        sensitivity = tp/(tp+fn+eps)
        specificity = tn /(fp+tn+eps)
        balanced_accuracy = (sensitivity + specificity)/2
        print("tp: "+str(tp)+", fp: "+str(fp)+", tn: "+str(tn)+", fn: "+str(fn))


if __name__ == "__main__":
    main()
