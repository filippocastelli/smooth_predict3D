from smooth_tiled_predictions import predict_img_with_smooth_windowing
from matplotlib import pylab as plt
from pathlib import Path
from skimage import io
import numpy as np
#%%
stack_path = Path("/media/phil/HDD0/dataset3d/orig_stacks")

try:
    mask_path
    img_path
except:
    img_path = stack_path.joinpath("training.tif")
    mask_path = stack_path.joinpath("training_groundtruth.tif")
    img_volume = io.imread(img_path, plugin = "pil")
    # mask_volume = io.imread(mask_path, plugin = "pil")
    # mask_volume = np.where(mask_volume == 255, 1, 0)
    
    

input_volume = img_volume[0:82, 0:384, 0:512]
    
# input_volume = np.zeros_like(input_volume) + 255
    
def predict_mock(img):
    return img/255


predictions_smooth = predict_img_with_smooth_windowing(
    np.expand_dims(input_volume, axis = -1),
    window_size=64,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    nb_classes=1,
    pred_func=(
        lambda img_batch_subdiv: predict_mock(img_batch_subdiv)),
    load = True
    )

