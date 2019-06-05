# MIT License
# Copyright (c) 2017 Vooban Inc.
# Coded by: Guillaume Chevalier
# Source to original code and license:
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches/blob/master/LICENSE


"""Do smooth predictions on an image from tiled prediction patches."""


import numpy as np
import scipy.signal
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import logging
logging.basicConfig(level=logging.DEBUG)
from pathlib import Path
import pickle

tmp_path = Path("tmp")
tmp_path.mkdir(exist_ok = True)


if __name__ == '__main__':
    PLOT_PROGRESS = True
    # See end of file for the rest of the __main__.
else:
    PLOT_PROGRESS = False


def debug_plt(image, idx = 0):
    plt.imshow(image[idx, :, :, 0])
    plt.show()
    plt.pause(1)
    
    
def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def _gaus_window(window_size, k = 2.608):

    x = np.arange(window_size)
    
    def gaus(x, x0, k):
        std = np.sqrt( x0**2 / (8*k))
        a = 2 / (1 - np.exp(-k))
        c = 2 - a
        y = a*np.exp(-((x-(x0/2))**2)/(2*std**2)) + c
        
        return y
    
    window = gaus(x, len(x), k)
    
    return window
    

cached_3d_windows = dict()
def _window_3D(window_size, mode = "gaus", power=2, k = 2.608):
    """
    _WINDOW_3D
    creates 3D window, either with a 3D gaussian or a 3d square spline

    Args:
        window_size (int): size of window.
        mode (string, optional): Mode of 3D window, can be "gaussian" or "spline". Defaults to "gaus".
        power (int, optional): Power for spline. Defaults to 2.
        k (float, optional): Parameter for gaussian spline. Defaults to 2.608.

    Raises:
        ValueError: if mode is unknown.

    Returns:
        wind (ndarray): 3d window.

    """
    global cached_3d_windows
    # key = "{}_{}".format(window_size, power)
    key = "{}_{}_{}".format(mode, window_size, k)
    if key in cached_3d_windows:
        wind = cached_3d_windows[key]
    else:
        
        if mode == "gaus":
             wind = _gaus_window(window_size, k)
        elif mode == "spline":
            wind = _spline_window(window_size, power)
        else:
            raise ValueError

        wind = np.expand_dims(wind, axis = -1)
        wind = np.expand_dims(wind, axis = -1)
        wind = np.expand_dims(wind, axis = -1)
        
        wind = wind * wind.transpose(1, 0, 2, 3) * wind.transpose(2,1,0, 3)
        
        wind = wind/wind.max()
        # profile3d = wind3d[32, 32, :, 0]
        # plt.plot(profile3d)
        # plt.show()

        if PLOT_PROGRESS:
            # For demo purpose, let's look once at the window:
            plt.imshow(wind[32, :, :, 0], cmap="viridis")
            plt.title("2D Section of 3D windowing function for a Smooth Blending of "
                      "Overlapping Patches")
            plt.show()
        cached_3d_windows[key] = wind
    return wind

def _pad_img(img, window_size, subdivisions):
    """
    Add borders to img for a "valid" border pattern according to "window_size" and
    "subdivisions".
    Image is an np array of shape (x, y, z, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (aug, aug), (0,0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    # gc.collect()

    # if PLOT_PROGRESS:
    #     # For demo purpose, let's look once at the window:
    #     plt.imshow(ret)
    #     plt.title("Padded Image for Using Tiled Prediction Patches\n"
    #               "(notice the reflection effect on the padded borders)")
    #     plt.show()
    return ret


def _unpad_img(padded_img, window_size, subdivisions):
    """
    Undo what's done in the `_pad_img` function.
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[
        aug:-aug,
        aug:-aug,
        :
    ]
    # gc.collect()
    return ret


def _rotate_mirror_do(im):
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.

    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    axes = ((0,1), (0,2), (1,2))
    
    rotations = []
    def _rotations(im, flip):
        logging.info("executing rotation set {}".format(flip))
        for i, ax in enumerate(tqdm(axes)):
            for n_rot in range(4):
                id_tuple = [flip, i, n_rot]
                fpath = tmp_path.joinpath("rot_{}_{}_{}.npy".format(flip,i, n_rot))
                np.save(file = fpath,
                        arr = np.rot90(np.array(im), k = n_rot, axes = ax))
                logging.debug("saving {}".format(fpath.name))
                rotations.append((id_tuple, fpath))
    
    _rotations(im, 0)
    im = np.array(im)[:, :, ::-1]
    _rotations(im, 1)
    im = np.array(im)[:, ::-1, :]
    _rotations(im, 2)

    rotpath = tmp_path.joinpath("rotpaths.pkl")
    
    with rotpath.open(mode = "wb") as rfile:
        pickle.dump(rotations, rfile)
        
    return rotations


def _rotate_mirror_undo(im_mirrs):
    """
    merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.

    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
    origs.append(np.array(im_mirrs[4])[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
    return np.mean(origs, axis=0)

class prediction_img():
    def __init__(self, padded_img, pred_func, window, subdivisions, rot):
        
        self.padded_img = padded_img
        self.pred_func = pred_func
        self.window = window
        self.window_size = window.shape[0]
        self.subdivisions = subdivisions
        self.rotation = rot
        self.aug = int(round(self.window_size * (1 - 1.0/subdivisions)))

        self.axes = ((0,1), (0,2), (1,2))
        # self.flips = ((0,0,0), (0,0,1), (0,1,1))
        self.flips = ((), (2), (1,2))
        
        self.pred_img = np.zeros_like(padded_img).astype("float")
        
    def _predict_batch(self, batchlist):
        batch_l = []
        pos_l = []
        for pos, patch in batchlist:
            patch = patch * self.window
            pos_l.append(pos)
            batch_l.append(patch)
            
        batch = np.array(batch_l)
        
        #(10,64,64,64,1)
        pred_batch = self.pred_func(batch)
        window_size = self.window_size
        for i, prediction in enumerate(pred_batch):
            z,y,x = pos_l[i]
            self.pred_img[z: z+window_size, y: y+window_size, x:x+window_size] = self.pred_img[z: z+window_size, y: y+window_size, x:x+window_size] + prediction 
        
        gc.collect()
        
    def _normalize_prediction(self, subdivisions = None):
        if subdivisions is None:
            subdivisions = self.subdivisions
        
        self.pred_img = self.pred_img / (subdivisions **2)
        
    def _back_transform(self):

        flipn, ax, k = self.rot
        #rot first, flip last
        self._unrot(flipn)
        self._unflip(ax,k)
        
    def _unflip(self, flipn):
        flip = self.flips[flipn]
        self.pred_img = np.flip(self.pred_img,axis = flip)
        
    def _unrot(self, ax, k):
        self.pred_img = np.rot90(self.pred_img,k = -k, axes = self.axes[ax])
        
    def _unpad(self):
        aug = self.aug
        self.pred_img = self.pred_img[aug:-aug, aug:-aug, aug:-aug, :]
        
    def plot_padded(self, idx = 0):
        plt.imshow(self.padded_img[idx, :, :,0])
        plt.colorbar()
        plt.show()
        
    def plot_pred(self, idx = 0):
        plt.imshow(self.pred_img[idx, :, :,0])
        plt.colorbar()
        plt.show()
        
        

    
def _windowed_subdivs(pad, window_size, subdivisions, nb_classes, pred_func):
    """
    Create tiled overlapping patches.

    Returns:
        5D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )

    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
    """
    padded_img_path = pad[1]
    rot = pad[0]
    
    #(images are in z,y,x,1 format)
    padded_img = np.load(padded_img_path)
    
    WINDOW = _window_3D(window_size=window_size, mode = "spline")
    
    pad_pred = prediction_img(padded_img, pred_func, WINDOW,subdivisions, rot)
    
    
    # padded_out_shape=list(padded_img.shape[:-1])+[nb_classes]
    
    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[2]
    pady_len = padded_img.shape[1]
    padz_len = padded_img.shape[0]
    
    subdivs = []
    
    x_points = range(0, padx_len-window_size+1, step)
    y_points = range(0, pady_len-window_size+1, step)
    z_points = range(0, padz_len-window_size+1, step)
    
    batchlist =  []
    max_batch = 10
    
    
    for z in z_points:
        # subdivs.append([])
        for y in y_points:
            # subdivs[-1].append([])
            for x in x_points:
                
                start_point = (z,y,x)
                patch = padded_img[z: z+window_size, y: y+window_size, x:x+window_size, :]
                # subdivs[-1][-1].append(patch)
                
                if len(batchlist) < max_batch-1:
                    batchlist.append((start_point, patch))
                else:
                    batchlist.append((start_point, patch))
                    pad_pred._predict_batch(batchlist)
                    batchlist = []
    
    if len(batchlist) > 0:
        pad_pred._predict_batch(batchlist)
        
    pad_pred.plot_padded()
    pad_pred.plot_pred()
    
    pad_pred._normalize_prediction()
    
    pad_pred.plot_padded()
    pad_pred.plot_pred()
    
    print("cucc")
    return subdivs


def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly.
    """
    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]

    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len-window_size+1, step):
        b = 0
        for j in range(0, pady_len-window_size+1, step):
            windowed_patch = subdivs[a, b]
            y[i:i+window_size, j:j+window_size] = y[i:i+window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1
    return y / (subdivisions ** 2)


def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func, load = True ):
    """
    Apply the `pred_func` function to square patches of the image, and overlap
    the predictions to merge them smoothly.

    See 6th, 7th and 8th idea here:
    http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/
    """
    pad = _pad_img(input_img, window_size, subdivisions)

    if load == True:
        rpath = tmp_path.joinpath("rotpaths.pkl")
        with rpath.open(mode = "rb") as rfile:
            pads = pickle.load(rfile)
            
    else:
        pads = _rotate_mirror_do(pad)
        
    
    # Note that the implementation could be more memory-efficient by merging
    # the behavior of `_windowed_subdivs` and `_recreate_from_subdivs` into
    # one loop doing in-place assignments to the new image matrix, rather than
    # using a temporary 5D array.

    # It would also be possible to allow different (and impure) window functions
    # that might not tile well. Adding their weighting to another matrix could
    # be done to later normalize the predictions correctly by dividing the whole
    # reconstructed thing by this matrix of weightings - to normalize things
    # back from an impure windowing function that would have badly weighted
    # windows.

    # For example, since the U-net of Kaggle's DSTL satellite imagery feature
    # prediction challenge's 3rd place winners use a different window size for
    # the input and output of the neural net's patches predictions, it would be
    # possible to fake a full-size window which would in fact just have a narrow
    # non-zero dommain. This may require to augment the `subdivisions` argument
    # to 4 rather than 2.

    res = []
    for pad in tqdm(pads):
        # For every rotation:
        sd = _windowed_subdivs(pad, window_size, subdivisions, nb_classes, pred_func)
        one_padded_result = _recreate_from_subdivs(
            sd, window_size, subdivisions,
            padded_out_shape=list(pad.shape[:-1])+[nb_classes])

        res.append(one_padded_result)

    # Merge after rotations:
    padded_results = _rotate_mirror_undo(res)

    prd = _unpad_img(padded_results, window_size, subdivisions)

    prd = prd[:input_img.shape[0], :input_img.shape[1], :]

    if PLOT_PROGRESS:
        plt.imshow(prd)
        plt.title("Smoothly Merged Patches that were Tiled Tighter")
        plt.show()
    return prd


def cheap_tiling_prediction(img, window_size, nb_classes, pred_func):
    """
    Does predictions on an image without tiling.
    """
    original_shape = img.shape
    full_border = img.shape[0] + (window_size - (img.shape[0] % window_size))
    prd = np.zeros((full_border, full_border, nb_classes))
    tmp = np.zeros((full_border, full_border, original_shape[-1]))
    tmp[:original_shape[0], :original_shape[1], :] = img
    img = tmp
    print(img.shape, tmp.shape, prd.shape)
    for i in tqdm(range(0, prd.shape[0], window_size)):
        for j in range(0, prd.shape[0], window_size):
            im = img[i:i+window_size, j:j+window_size]
            prd[i:i+window_size, j:j+window_size] = pred_func([im])
    prd = prd[:original_shape[0], :original_shape[1]]
    if PLOT_PROGRESS:
        plt.imshow(prd)
        plt.title("Cheaply Merged Patches")
        plt.show()
    return prd


def get_dummy_img(xy_size=128, nb_channels=3):
    """
    Create a random image with different luminosity in the corners.

    Returns an array of shape (xy_size, xy_size, nb_channels).
    """
    x = np.random.random((xy_size, xy_size, nb_channels))
    x = x + np.ones((xy_size, xy_size, 1))
    lin = np.expand_dims(
        np.expand_dims(
            np.linspace(0, 1, xy_size),
            nb_channels),
        nb_channels)
    x = x * lin
    x = x * lin.transpose(1, 0, 2)
    x = x + x[::-1, ::-1, :]
    x = x - np.min(x)
    x = x / np.max(x) / 2
    gc.collect()
    if PLOT_PROGRESS:
        plt.imshow(x)
        plt.title("Random image for a test")
        plt.show()
    return x


def round_predictions(prd, nb_channels_out, thresholds):
    """
    From a threshold list `thresholds` containing one threshold per output
    channel for comparison, the predictions are converted to a binary mask.
    """
    assert (nb_channels_out == len(thresholds))
    prd = np.array(prd)
    for i in range(nb_channels_out):
        # Per-pixel and per-channel comparison on a threshold to
        # binarize prediction masks:
        prd[:, :, i] = prd[:, :, i] > thresholds[i]
    return prd


if __name__ == '__main__':
    ###
    # Image:
    ###

    img_resolution = 600
    # 3 such as RGB, but there could be more in other cases:
    nb_channels_in = 3

    # Get an image
    input_img = get_dummy_img(img_resolution, nb_channels_in)
    # Normally, preprocess the image for input in the neural net:
    # input_img = to_neural_input(input_img)

    ###
    # Neural Net predictions params:
    ###

    # Number of output channels. E.g. a U-Net may output 10 classes, per pixel:
    nb_channels_out = 3
    # U-Net's receptive field border size, it does not absolutely
    # need to be a divisor of "img_resolution":
    window_size = 128

    # This here would be the neural network's predict function, to used below:
    def predict_for_patches(small_img_patches):
        """
        Apply prediction on images arranged in a 4D array as a batch.

        Here, we use a random color filter for each patch so as to see how it
        will blend.

        Note that the np array shape of "small_img_patches" is:
            (nb_images, x, y, nb_channels_in)
        The returned arra should be of the same shape, except for the last
        dimension which will go from nb_channels_in to nb_channels_out
        """
        small_img_patches = np.array(small_img_patches)
        rand_channel_color = np.random.random(size=(
            small_img_patches.shape[0],
            1,
            1,
            small_img_patches.shape[-1])
        )
        return small_img_patches * rand_channel_color * 2

    ###
    # Doing cheap tiled prediction:
    ###

    # Predictions, blending the patches:
    cheaply_predicted_img = cheap_tiling_prediction(
        input_img, window_size, nb_channels_out, pred_func=predict_for_patches
    )

    ###
    # Doing smooth tiled prediction:
    ###

    # The amount of overlap (extra tiling) between windows. A power of 2, and is >= 2:
    subdivisions = 2

    # Predictions, blending the patches:
    smoothly_predicted_img = predict_img_with_smooth_windowing(
        input_img, window_size, subdivisions,
        nb_classes=nb_channels_out, pred_func=predict_for_patches
    )

    ###
    # Demonstrating that the reconstruction is correct:
    ###

    # No more plots from now on
    PLOT_PROGRESS = False

    # useful stats to get a feel on how high will be the error relatively
    print(
        "Image's min and max pixels' color values:",
        np.min(input_img),
        np.max(input_img))

    # First, defining a prediction function that just returns the patch without
    # any modification:
    def predict_same(small_img_patches):
        """
        Apply NO prediction on images arranged in a 4D array as a batch.
        This implies that nb_channels_in == nb_channels_out: dimensions
        and contained values are unchanged.
        """
        return small_img_patches

    same_image_reconstructed = predict_img_with_smooth_windowing(
        input_img, window_size, subdivisions,
        nb_classes=nb_channels_out, pred_func=predict_same
    )

    diff = np.mean(np.abs(same_image_reconstructed - input_img))
    print(
        "Mean absolute reconstruction difference on pixels' color values:",
        diff)
    print(
        "Relative absolute mean error on pixels' color values:",
        100*diff/(np.max(input_img)) - np.min(input_img),
        "%")
    print(
        "A low error (e.g.: 0.28 %) confirms that the image is still "
        "the same before and after reconstruction if no changes are "
        "made by the passed prediction function.")
