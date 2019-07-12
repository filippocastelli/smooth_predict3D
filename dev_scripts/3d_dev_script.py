import sys
import numpy as np
from skimage import io
sys.path.append("..")
from smooth_predict3D import smooth_predict, debug_plt
#%%

volume = io.imread("/media/phil/HDD0/dataset3d/orig_stacks/training.tif")
# gt = io.imread("/media/phil/HDD0/dataset3D/orig_stacks/training_groundtruth.tif")

volume_prep  = np.expand_dims(volume, -1)
debug_plt(volume_prep, title = "original")
def pred_mock(image):
    return image


sp = smooth_predict(input_img = volume_prep,
                    window_size = (64,64,64),
                    subdivisions = 2,
                    nb_classes = 1,
                    pred_func = pred_mock,
                    max_batch = 10,
                    rot_axes = ((1,2),),
                    flip_axes =[(1,), (2,)],
                    load = True)