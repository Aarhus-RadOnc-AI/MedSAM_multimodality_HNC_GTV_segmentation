import numpy as np
import os 
import glob
import json

import os
import pickle
import json
from typing import List
import SimpleITK as sitk
"""
REFERENCE FROM https://github.com/MIC-DKFZ/batchgenerators/

"""
def subdirs(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def nifti_files(folder: str, join: bool = True, sort: bool = True) -> List[str]:
    return subfiles(folder, join=join, sort=sort, suffix='.nii.gz')


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)


def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def pardir(path: str):
    return os.path.join(path, os.pardir)


def split_path(path: str) -> List[str]:
    """
    splits at each separator. This is different from os.path.split which only splits at last separator
    """
    return path.split(os.sep)



def reconstruct_seg_df_from_json(path):
    sum_path = os.path.join(path, 'summary.json')

    with open(sum_path) as file:
        json_dict = json.load(file)
        
    score_dict1 = {}
    score_dict2 = {}
    metrics = json_dict['results']['all'][0]['1'].keys()
    
    for i,data in enumerate(json_dict['results']['all']):
            base_name = os.path.basename(json_dict['results']['all'][i]['reference'])
            pt_name = base_name.replace(".nii.gz", "")
            #patient_list.append(pt_name)
            score_dict1[pt_name] = dict()
            #score_dict2[pt_name] = dict()
            #score_dict1[pt_name]['PatientID'] = pt_name
            for metric in metrics:
                if metric != 'volume_diff':
                    if metric == 'Dice':
                        score_dict1[pt_name]['DSC'] = data['1'][metric]
                    elif metric == 'Hausdorff Distance 95':
                        score_dict1[pt_name]['HD95 (mm)'] = data['1'][metric]
                    elif metric == 'Avg. Surface Distance':
                        score_dict1[pt_name]['Mean Surface Distance (mm)'] = data['1'][metric]
                    elif metric == 'Total Positives Reference':
                        score_dict1[pt_name]['Volume (cc)'] = np.round(data['1'][metric]/1000,2)
                    elif metric == 'Total Positives Test':
                        score_dict1[pt_name]['Pred Volume (cc)'] = np.round(data['1'][metric]/1000,2)

                    else:
                        score_dict1[pt_name][metric] = data['1'][metric]
            if score_dict1[pt_name]['Volume (cc)'] == 0:
                if score_dict1[pt_name]['Pred Volume (cc)'] == 0:
                    #print(score_dict1[pt_name]['Pred Volume (cc)'])
                    score_dict1[pt_name]['DSC'] = np.nan
                    score_dict1[pt_name]['HD95 (mm)'] = np.nan
                    score_dict1[pt_name]['Mean Surface Distance (mm)'] = np.nan
                    score_dict1[pt_name]['False Discovery Rate']= np.nan
                    score_dict1[pt_name]['False Negative Rate']= np.nan
                    score_dict1[pt_name]['False Omission Rate']= np.nan
                    score_dict1[pt_name]['False Positive Rate']= np.nan
                    score_dict1[pt_name]['Surface Dice 2mm']= np.nan
                    score_dict1[pt_name]['Surface Dice 3mm']= np.nan

            score_dict1[pt_name]['GTV'] = 'GTV-T'
                        
                        
    for i,data in enumerate(json_dict['results']['all']):
            base_name = os.path.basename(json_dict['results']['all'][i]['reference'])
            pt_name = base_name.replace(".nii.gz", "")
            #patient_list.append(pt_name)
            score_dict2[pt_name] = dict()
            #score_dict2[pt_name] = dict()
            #score_dict2[pt_name]['PatientID'] = pt_name
            for metric in metrics:
                    if metric == 'Dice':
                        score_dict2[pt_name]['DSC'] = data['2'][metric]
                    elif metric == 'Hausdorff Distance 95':
                        score_dict2[pt_name]['HD95 (mm)'] = data['2'][metric]
                    elif metric == 'Avg. Surface Distance':
                        score_dict2[pt_name]['Mean Surface Distance (mm)'] = data['2'][metric]
                    elif metric == 'Total Positives Reference':
                        score_dict2[pt_name]['Volume (cc)'] = np.round(data['2'][metric]/1000,2)
                    elif metric == 'Total Positives Test':
                        score_dict2[pt_name]['Pred Volume (cc)'] = np.round(data['2'][metric]/1000,2)
                    else:
                        score_dict2[pt_name][metric] = data['2'][metric]

            if score_dict2[pt_name]['Volume (cc)'] == 0:
                if score_dict2[pt_name]['Pred Volume (cc)'] == 0:
                    #print(score_dict2[pt_name]['Pred Volume (cc)'])
                    score_dict2[pt_name]['DSC'] = np.nan
                    score_dict2[pt_name]['HD95 (mm)'] = np.nan
                    score_dict2[pt_name]['Mean Surface Distance (mm)'] = np.nan
                    score_dict2[pt_name]['False Discovery Rate']= np.nan
                    score_dict2[pt_name]['False Negative Rate']= np.nan
                    score_dict2[pt_name]['False Omission Rate']= np.nan
                    score_dict2[pt_name]['False Positive Rate']= np.nan
                    score_dict2[pt_name]['Surface Dice 2mm']= np.nan
                    score_dict2[pt_name]['Surface Dice 3mm']= np.nan

            score_dict2[pt_name]['GTV'] = 'GTV-N'

    return score_dict1, score_dict2


## Summary for calibration confidence scores
def reconstruct_calib_df_from_json(path):
    sum_path = os.path.join(path, 'calib_summary.json')
    with open(sum_path) as file:
        json_dict = json.load(file)
        
    score_dict1 = {}
    score_dict2 = {}
    metrics = json_dict['results']['all'][0]['1'].keys()
    
    for i,data in enumerate(json_dict['results']['all']):
            base_name = os.path.basename(json_dict['results']['all'][i]['reference'])
            pt_name = base_name.replace(".nii.gz", "")
            #patient_list.append(pt_name)
            score_dict1[pt_name] = dict()

            for metric in metrics:
                    score_dict1[pt_name][metric] = data['1'][metric]
            #score_dict1[pt_name]['GTV'] = 0
                        
                        
    for i,data in enumerate(json_dict['results']['all']):
            base_name = os.path.basename(json_dict['results']['all'][i]['reference'])
            pt_name = base_name.replace(".nii.gz", "")
            #patient_list.append(pt_name)
            score_dict2[pt_name] = dict()

            for metric in metrics:
                score_dict2[pt_name][metric] = data['2'][metric]
            #score_dict2[pt_name]['GTV'] = 1

    return score_dict1, score_dict2
    
def reconstruct_UED_df_from_json(path):
    path = path+ '_union'
    sum_path = os.path.join(path, 'summary.json')
    print(sum_path)
    with open(sum_path) as file:
        json_dict = json.load(file)
        
    score_dict1 = {}

    for i,data in enumerate(json_dict['results']['all']):
            base_name = os.path.basename(json_dict['results']['all'][i]['reference'])
            pt_name = base_name.replace(".nii.gz", "")
            #patient_list.append(pt_name)
            score_dict1[pt_name] = dict()
            score_dict1[pt_name]['UED'] = data['1']['Dice']                      
                        
    return score_dict1

def calcualte_target_entropy_with_th(arguments, only_seg_roi=False):
    #th = 0.7
    
    umap_folder , seg_folder,  filename, th= arguments

    # create an dict for data storage

    seg_path = join(seg_folder, filename) 
    umap_path = join(umap_folder, filename.replace(".nii.gz", ".npz")) 
    pid = filename.replace(".nii.gz", "")
    
    
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    umap = np.load(umap_path)['umap']

    resutl_list = []

    for target in range(1,3):

        entropy_results = {}
        entropy_results['PatientID'] = pid

        if target == 1:
            gtv = 'GTV-T'
            
        elif target == 2:
            gtv = 'GTV-N'

        # create an dict for each GTV
        entropy_results['GTV'] = gtv
    
        mask_seg = seg==target

        target_umap = umap[target]
        mask_umap = target_umap>th

        
        roi = mask_seg

        if not only_seg_roi:
            if np.sum(mask_seg) != 0:
                union_roi = (mask_seg + mask_umap)
                union_roi[union_roi>0] = 1
                roi = union_roi
        

        seg_entropy = target_umap * roi
        seg_entropy = seg_entropy[seg_entropy> 0] 



        if np.sum(mask_seg) == 0:
            # no segmentation made for this target (mostly GTV-N)
            entropy_results["Total Entropy"] = np.nan 
            entropy_results["Mean Entropy"] = np.nan 
            entropy_results["Entropy STD"] = np.nan 
            entropy_results["Entropy Volume"] = np.nan 
            entropy_results["Entropy Coefficient of Variation"] = np.nan 
            entropy_results["Logarithm Entropy Coefficient of Variation"] = np.nan 
            # entropy_results["Uncertainty-Segmentation Hausdorff distance"] = np.nan 
            # entropy_results["Uncertainty-Segmentation Mean Surface Distance"] = np.nan 

        else:
            # eros_seg  = binary_erosion(mask_seg, iterations=2).astype(int)
            # seg_line = mask_seg.astype(int) - eros_seg
            # e_hd  = hd(mask_umap, seg_line)
            # e_asd = asd(mask_umap, seg_line)
            entropy_mean = np.mean(seg_entropy) 
            entropy_std = np.std(seg_entropy) 
            entropy_results["Total Entropy"] = np.sum(seg_entropy)
            entropy_results["Mean Entropy"] = entropy_mean
            entropy_results["Entropy STD"] = entropy_std
            entropy_results["Entropy Volume"] = np.sum(roi) 
            entropy_results["Entropy Coefficient of Variation"] = entropy_std / (entropy_mean  + 1e-6) # + 1e-6 to avoid divide by 0
            entropy_results["Logarithm Entropy Coefficient of Variation"] = np.log(entropy_std / (entropy_mean  + 1e-6)) 
            # entropy_results["Uncertainty-Segmentation Hausdorff distance"] = e_hd
            # entropy_results["Uncertainty-Segmentation Mean Surface Distance"] = e_asd

        resutl_list.append(entropy_results)

    return resutl_list

join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
listdir = os.listdir
makedirs = maybe_mkdir_p
os_split_path = os.path.split
