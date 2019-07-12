# =============================================================================
# smooth_predict3D
#
# Coded by FIlippo Maria Castelli
# 
# Smooth 3D patch-based predictions for extended volumes
#
# forked from 2D project Smoothly-Blend-Image-Patches by Guillaume Chevalier
# https://github.com/Vooban/Smoothly-Blend-Image-Patches
# =============================================================================

# IMPORTS
import numpy as np
import scipy.signal
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
import pickle


#TODO: Optimize data views

class smooth_predict():
    """
    smooth_predict
    
    Patch-based prediction of extended 3D volumes:
    Takes the original volume, makes 3D rotations and flips, performs patch-based
    reconstruction of prediction using a 3D square spline window function and then
    averages the back-transformed results to output a final prediction volume.
    
    smooth_predict is agnostic to the particular system used to generate predictions
    so it's intrinsecally compatible with any model from Keras, TF, sklearn, Pytorch etc...
    
    note: smooth_predict generates a temp folder for storage of heavy tensors to free
    up RAM space: make sure there is enough disk-space available for the script execution,
    current setup needs at least 37x the dimension of the original 3D volume
    on disk and 3x on RAM.
    
    note: smooth_predict operates in channel_last convention, however you can
    pre-transorm input data to be in channel_last format and define a lambda for 
    the prediction conversion, format selection can be part of a future release
    
    note: current padding scheme needs the window_size parameter to be an even 
    multiple of the subdivisions parameter, this limitation can be overcome in a
    future release.
    
    Parameters
    ----------
    input_img : ndarray
        input tensor, should be in the format (z,y,x,channels),
        first three axes should be permutable so there's no real need for 
        a specific ordering as long as channel_last format for input data is mantained
        If your model operates in channel_first format see pred_func
    window_size : int
        Linear size of the cubic window, must be an even multiple of subdivisions.
    subdivisions : int
        Number of times each pixel has to be predicted in extended image reconstruction,
        must be an even number, typically 2.
    nb_classes : int
        Number of non-background classes.
    pred_func : lambda
        Prediction function, you can just write a lambda that returns the prediction
        results for a given input batch.
        
        >>> lambda pred_func : image_batch : model.predict(image_batch)
    max_batch : int, optional
        Max number of items to be sent in a prediction batch,
        choose it accordingly to your GPU and your model. The default is 10.
    tmp : str, optional
        name of temp folder. The default is "tmp".
    window_mode : str, optional
        INTENDED AS A DEBUG TOOL ONLY, type of 3D window function, can be
        "spline" or "gaus". The default is "spline".
    load : bool, optional
        INTENDED AS A DEBUG TOOL ONLY, loads pre-existend tmp files instead of creating
        new ones. The default is False.
    rot_axes : tuple
        specify rotation axes planes
    

    Returns
    -------
    out_img : ndarray
        output tensor

    """
    def __init__(self, input_img,
                 window_size,
                 subdivisions,
                 nb_classes,
                 pred_func,
                 max_batch = 10,
                 tmp = "tmp",
                 window_mode = "spline",
                 rot_axes = [(0,1), (0,2), (1,2)],
                 flip_axes = [(1,), (2,)],
                 load = False,
                 debug_plots = False):

        #SANITY CHECKS
        assert len(input_img.shape) == 4; "Data must be in z,y,x,channels format"
        for rot_couple in rot_axes:
            assert len(rot_couple) == 2; "Two axes are required for defining a rotation plane"
            assert all(dim < 4 for dim in rot_couple); "Rotation plane does not exist"
            assert rot_couple[0] != rot_couple[1]; "Rotation plane must be defined by two different axes"

        for ax in flip_axes:
            assert len(ax) < 3
            assert all(dim <  len(input_img.shape) for dim in ax)

        self.tmp_path = Path(tmp)
        self.tmp_path.mkdir(exist_ok = True)
        
        #INPUTS
        self.input_img = input_img
        self.window_size = window_size
        self.subdivisions = subdivisions
        self.nb_classes = nb_classes
        self.pred_func = pred_func
        self.load_flag = load
        self.window_mode = window_mode
        self.load = load
        self.max_batch = max_batch
        self.debug_plots = debug_plots

        # OTHER ATTRIBUTES
        self.rot_axes = rot_axes
        self.flip_axes = flip_axes
        self.flip_axes.insert(0, ())
        self.rotations = []
        self.cached_windows = dict()
        
        # INIT SEQUENCE
        
        #padding original img
        self.padded_original, self.padding = self._pad_img(in_img = self.input_img)
        
        self.padded_out_shape=list(self.padded_original.shape[:-1])+[self.nb_classes]
        self.out_shape = list(self.input_img.shape[:-1])+[self.nb_classes]
        
        #generate rotations
        # debug only, if load == True load previously saved files
        if self.load == True:
            self.rotations = self._load_tmp()
        else:
            self.rotations = self._gen_rotations(self.padded_original)
        
        self.window = self.window_3D(mode = self.window_mode)
        
        self.out_img = np.zeros(shape = self.out_shape, dtype = "float")
        
        self._average_predicted_views()
        
        
    def _pad_img(self, in_img):
        """
        _pad_img
        
        applies padding to input img

        Parameters
        ----------
        in_img : ndarray
            input img.

        Returns
        -------
        padded_img : ndarray
            padded img.
        pads : TYPE
            list of padding parameters,
            [[in_pad_z, out_pad_z],[in_pad_y, out_pad_y],[in_pad_x, out_pad_x], [0,0]] format.

        """
        
        assert self.window_size[0] % self.subdivisions == 0; "window size must be divisible by subdivisions"
        
        aug_unit = int(self.window_size[0]/self.subdivisions)
        
        dims = in_img.shape[:-1]
        # half_window = int(self.window_size / 2)
        
        pads = np.array([[0,0], [0,0], [0,0], [0,0]])
        
        for i, dim in enumerate(dims):
            pads[i] = pads[i] + aug_unit
            dim = dim + pads[i].sum()
            r = dim%aug_unit
            pads[i][0] = pads[i][0] + r // 2
            pads[i][1] = pads[i][1] + r // 2 + r%2

        padded_img = np.pad(in_img, pad_width=pads, mode='reflect')
        
        return padded_img, pads
    
    def _load_tmp(self):
        """ loads self.rotations from disk"""
        rpath = self.tmp_path.joinpath("rotpaths.pkl")
        
        with rpath.open(mode = "rb") as rfile:
            rotations = pickle.load(rfile)
            
        return rotations
    
    def _gen_rotations(self, padded_img):
        """
        _gen_rotations
        
        Routine for generating flipped and rotation version of the input, saves
        the results in temp folders and appends rotation_id and path to saved file
        in 'rotations'.
        
        Parameters
        ----------
        padded_img : ndarray
            input img.

        Returns
        -------
        rotations : list
            [rotation_id, Path] format.
            rotation_id is a tuple (flip_axis, rot_axis, n_of_rotations)
            flip_axis is an index of self.flip_axes
            rot_axis is an index of self.rot_axes
            n_of_rotations is an int
            Path is a pathlib-style path
        """
        
        img = padded_img
        r_list = []
        
        logging.info("Executing flips and rotations of original dataset")
        for i, flip in enumerate(tqdm(self.flip_axes)):
            #flip img vector
            img = np.flip(img, axis = flip)
            r_list = self._rot_save(img, flip, r_list)

        rotpath = self.tmp_path.joinpath("rotpaths.pkl")
        
        with rotpath.open(mode = "wb") as rfile:
            pickle.dump(r_list, rfile)
            
        return r_list

    def _rot_save(self, im, flip, r_list):
        """ method for generating rotations of already flipped tensors and saving
        them"""
        logging.debug("Executing rotation set {}".format(flip))
        for i, ax in enumerate(tqdm(self.rot_axes)):
            for n_rot in range(3):
                id_tuple = [flip, i, n_rot]
                fpath = self.tmp_path.joinpath("rot_{}_{}_{}.npy".format(''.join(tuple(map(str, flip))),i, n_rot))
                np.save(file = fpath,
                        arr = np.rot90(np.array(im), k = n_rot, axes = ax))
                logging.debug("saving {}".format(fpath.name))
                r_list.append((id_tuple, fpath))
        
        return r_list
                
    def window_3D(self, mode = "spline", power=2, k = 2.608):
        """
        generates 3D window function from 1D profiles

        Parameters
        ----------
        mode : str, optional
            Can be "spline" or "gaus". The default is "spline".
        power : int, optional
            For spline function, power of spline approx. The default is 2.
        k : TYPE, optional
            For gaus function, ratio of x0/sigma. The default is 2.608.

        Raises
        ------
        ValueError
            if mode is unknown.

        Returns
        -------
        wind : ndarray
            (window_size, window_size, window_size) window function.

        """
        
        key = "{}_{}_{}".format(mode, self.window_size, k)
        
        if key in self.cached_windows:
            wind = self.cached_windows[key]
        else:
            if mode == "gaus":
                logging.debug("shouldnt really be using that, possible artefacts")
                wind = self.gaus_window(self.window_size[0], k)
            elif mode == "spline":
                wind = self.spline_window(self.window_size[0], power)
            else:
                raise ValueError
    
            wind = np.expand_dims(wind, axis = -1)
            wind = np.expand_dims(wind, axis = -1)
            wind = np.expand_dims(wind, axis = -1)
            
            wind = wind * wind.transpose(1, 0, 2, 3) * wind.transpose(2,1,0, 3)
            
            wind = wind/wind.max()
            
            self.cached_windows[key] = wind
            
        return wind
    
    @classmethod
    def spline_window(cls, window_size, power=2):
        """ generates 1D spline window"""
        intersection = int(window_size/4)
        wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
        wind_outer[intersection:-intersection] = 0
    
        wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0
    
        wind = wind_inner + wind_outer
        wind = wind / np.average(wind)
        return wind

    @classmethod
    def gaus_window(cls, window_size, k = 2.608):
        """generates 1D gaus window"""
    
        x = np.arange(window_size[0])
        def gaus(x, x0, k):
            std = np.sqrt( x0**2 / (8*k))
            a = 2 / (1 - np.exp(-k))
            c = 2 - a
            y = a*np.exp(-((x-(x0/2))**2)/(2*std**2)) + c
            
            return y
        window = gaus(x, len(x), k)
        
        return window
     
    def _predict_view(self, rotation):
        """
        for a given flip/rotation of 3D input space instantiates a single_view_predictor
        for patch-based prediction of that view

        Parameters
        ----------
        rotation : list
            Tuple in format (rot_id, rot_path), see smooth_predict._gen_rotations
            docstrings for details.

        Returns
        -------
        predicted_view : ndarray
            reconstructed 3D prediction.

        """
        
        pred = single_view_predictor(rotation = rotation,
                                     pred_func = self.pred_func,
                                     window = self.window,
                                     subdivisions = self.subdivisions,
                                     max_batch = self.max_batch,
                                     rot_axes = self.rot_axes,
                                     flip_axes = self.flip_axes,
                                     padding = self.padding)
        
        predicted_view = pred.predict_from_patches()
        
        del pred
        gc.collect()
        
        return predicted_view
    
    def _average_predicted_views(self):
        """ performs an average of all views to obtain final prediction"""
        
        logging.info("Predicting and averaging single views")
        for rotation in tqdm(self.rotations):
            current_view = self._predict_view(rotation)
            if self.debug_plots == True:
                debug_plt(current_view, title = str(rotation[0]))
            self.out_img = self.out_img + current_view
        
        self.out_img = self.out_img/len(self.rotations)
        
        return self.out_img

class single_view_predictor():
    def __init__(self,
                 rotation,
                 pred_func,
                 window,
                 padding,
                 subdivisions,
                 max_batch = 8,
                 rot_axes = None,
                 flip_axes = None):
        """
        single_view_predictor class
        
        executes patch-based prediction of a single flipped/rotated view and then
        performs back-transform to the original format

        Parameters
        ----------
        rotation : tuple
            (rot_id, rot_path).
        pred_func : lambda
            single patch prediction function.
        window : ndarray
            3D window for patch masking.
        padding : list
            vector containing information on padding, see smooth_predict._pad_img
            docstrings for further informations.
        subdivisions : int
            Number of subdivisions.
        max_batch : int, optional
            Max items to put in a batch. The default is 8.
        rot_axes : list, optional
            list of rotation axes. The default is None.
        flip_axes : list, optional
            list of flipping axes. The default is None.

        Returns
        -------
        out_view : ndarray
            output prediction

        """
        
        #INPUTS
        
        self.path = rotation[1]
        self.rot_id = rotation[0]
        self.pred_func = pred_func
        self.window = window
        self.window_size = window.shape[0]
        self.subdivisions = subdivisions
        self.max_batch = max_batch
        self.padding = padding

        
        #OTHER ATTRIBUTES
        self.aug = int(round(self.window_size * (1 - 1.0/subdivisions)))
        
        
        if rot_axes is None:
            self.rot_axes = ((0,1), (0,2), (1,2))
        else:
            self.rot_axes = rot_axes
            
        if flip_axes is None:
            self.flip_axes = ((), (2), (1,2))
        else:
            self.flip_axes = flip_axes
        
        self.batch_queue = []
        
        #INIT SEQUENCE
        
        # load padded img
        self.padded_img = np.load(self.path)
        #create prediction blank
        self.pred_img = np.zeros_like(self.padded_img).astype("float")
        
    def _predict_batch(self):
        """ create a batch, send it to predict_func,
        multiply results by window function,
        add results to output image,
        clear batch
        """
        batch_l = []
        pos_l = []
        for pos, patch in self.batch_queue:
            pos_l.append(pos)
            batch_l.append(patch)
            
        batch = np.array(batch_l)
        
        #(10,64,64,64,1)
        pred_batch = self.pred_func(batch)
        window_size = self.window_size
        for i, prediction in enumerate(pred_batch):
            #TODO: speed up by extending window to batch size and multiplying entire batch
            #instead of doing it patch-per-patch
            prediction = prediction * self.window
            z,y,x = pos_l[i]
            self.pred_img[z: z+window_size, y: y+window_size, x:x+window_size] = self.pred_img[z: z+window_size, y: y+window_size, x:x+window_size] + prediction 
        
        #empty queue
        self.batch_queue = []
        gc.collect()
        
        return self
        
    def _normalize(self, subdivisions = None):
        """ divide image by subdivisions^3"""
        if subdivisions is None:
            subdivisions = self.subdivisions
        
        self.pred_img = self.pred_img / (subdivisions **3)
        
    def _back_transform(self):
        """perform unrotation, unflipping and unpadding"""

        flip_ax, ax, k = self.rot_id
        #rot first, flip last
        self._unrot(ax,k)
        self._unflip(flip_ax)
        self._unpad()
        
    def _unflip(self, flip_ax):
        """ flips image to original format"""
        self.pred_img = np.flip(self.pred_img,axis = flip_ax)
        
    def _unrot(self, ax, k):
        """rotate image to original format"""
        self.pred_img = np.rot90(self.pred_img,k = -k, axes = self.rot_axes[ax])
        
    def _unpad(self):
        """removes padding"""
        # aug = self.aug
        z_min, z_max = self.padding[0]
        y_min, y_max = self.padding[1]
        x_min, x_max = self.padding[2]
        
        self.pred_img = self.pred_img[z_min : -z_max, y_min : -y_max, x_min : -x_max, :]
        
    def plot_padded(self, idx = 0):
        """debug plot function to see padded image"""
        plt.imshow(self.padded_img[idx, :, :,0])
        plt.colorbar()
        plt.show()
        
    def plot_pred(self, idx = 0):
        """debug plot function to see predicted image"""
        plt.imshow(self.pred_img[idx, :, :,0])
        plt.colorbar()
        plt.show()
        
    def predict_from_patches(self):
        """Prediction routine: creates patches of original input volume, feeds them
        to predictor in batches, performs normalization and back transforming
        retunrs output image"""

        padz_len, pady_len, padx_len = self.padded_img.shape[:-1]
        
        step = int(self.window_size/self.subdivisions)
        
        x_points = range(0, padx_len-self.window_size+1, step)
        y_points = range(0, pady_len-self.window_size+1, step)
        z_points = range(0, padz_len-self.window_size+1, step)
   
        
        for z in tqdm(z_points):
            for y in y_points:
                for x in x_points:
                    start_point = (z,y,x)
                    patch = self.padded_img[z: z+self.window_size, y: y+self.window_size, x:x+self.window_size, :]
                    
                    assert patch.shape == (self.window_size,self.window_size,self.window_size, self.padded_img.shape[-1]);"Padded image should contain an integer number of windows, something's wrong with padding"
                    if len(self.batch_queue) < self.max_batch-1:
                        self.batch_queue.append((start_point, patch))
                    else:
                        self.batch_queue.append((start_point, patch))
                        self._predict_batch()
        
        #if there are remaining patches process them
        if len(self.batch_queue) > 0:
            self._predict_batch()

        self._normalize()
        self._back_transform()
        
        return self.pred_img

# =============================================================================
# misc
# =============================================================================
def debug_plt(image, idx = 0, title = None):
    """dirty function for debuggin in ipdb"""
    plt.imshow(image[idx, :, :, 0])
    if title is not None:
        plt.title(title)
    plt.show()
    plt.pause(1)

