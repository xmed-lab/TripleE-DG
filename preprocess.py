import os
import numpy as np
import glob
image_path = "/home/eexmli/datasets/PAH-prediction/data_pa_nii_2nd_center/*image.nii.gz"
label_path = "/home/eexmli/datasets/PAH-prediction/data_pa_nii_2nd_center/*label.nii.gz"
image_list = glob.glob(image_path)
label_list = glob.glob(label_path)
image_list = [item for item in image_list if "6_" not in item]
image_list = [item for item in image_list if "7_" not in item]
print (len(image_list))
print (len(label_list))
