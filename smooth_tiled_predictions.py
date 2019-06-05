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


def debug_plt(image, idx = 0):
    plt.imshow(image[idx, :, :, 0])
    plt.show()
    plt.pause(1)
    
class predictor():
    
    def __init__(self, input_img,
                 window_size,
                 subdivisions,
                 nb_classes,
                 pred_func,
                 max_batch = 10,
                 load = True,
                 window_mode = "spline",
                 tmp = "tmp"):
        
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
        
        # OTHER ATTRIBUTES
        self.rot_axes = ((0,1), (0,2), (1,2))
        self.flip_axes = ((), (2), (1,2))
        self.rotations = []
        self.cached_windows = dict()
        
        # INIT SEQUENCE
        
        #padding original img
        self.padded_original, self.padding = self.pad_img(in_img = self.input_img)
        
        self.padded_out_shape=list(self.padded_original.shape[:-1])+[self.nb_classes]
        self.out_shape = list(self.input_img.shape[:-1])+[self.nb_classes]
        
        #generate rotations
        # debug only, if load == True load previously saved files
        if self.load == True:
            self.rotations = self.load_tmp()
        else:
            self.rotations = self.gen_rotations(self.padded_original)
        
        self.window = self.window_3D(mode = self.window_mode)
        
        self.out_img = np.zeros(shape = self.out_shape, dtype = "float")
        
        self.average_predicted_views()
        
        
    def pad_img(self, in_img):
        
        assert self.window_size % self.subdivisions == 0; "window size must be divisible by subdivisions"
        
        aug_unit = int(self.window_size/self.subdivisions)
        
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
    
    def load_tmp(self):
        rpath = self.tmp_path.joinpath("rotpaths.pkl")
        
        with rpath.open(mode = "rb") as rfile:
            rotations = pickle.load(rfile)
            
        return rotations
    
    def gen_rotations(self, padded_img):
        
        img = padded_img
        r_list = []
        
        for i, flip in enumerate(self.flip_axes):
            #flip img vector
            img = np.flip(img, axis = flip)
            rotations = self._rot_save(img, i, r_list)

        rotpath = self.tmp_path.joinpath("rotpaths.pkl")
        
        with rotpath.open(mode = "wb") as rfile:
            pickle.dump(rotations, rfile)
            
            return rotations

    def _rot_save(self, im, flip, r_list):
        logging.info("executing rotation set {}".format(flip))
        for i, ax in enumerate(tqdm(self.rot_axes)):
            for n_rot in range(4):
                id_tuple = [flip, i, n_rot]
                fpath = self.tmp_path.joinpath("rot_{}_{}_{}.npy".format(flip,i, n_rot))
                np.save(file = fpath,
                        arr = np.rot90(np.array(im), k = n_rot, axes = ax))
                logging.debug("saving {}".format(fpath.name))
                r_list.append((id_tuple, fpath))
        
        return r_list
                
    def window_3D(self, mode = "spline", power=2, k = 2.608):
        
        key = "{}_{}_{}".format(mode, self.window_size, k)
        
        if key in self.cached_windows:
            wind = self.cached_windows[key]
        else:
            if mode == "gaus":
                logging.debug("shouldnt really be using that, possible artefacts")
                wind = self.gaus_window(self.window_size, k)
            elif mode == "spline":
                wind = self.spline_window(self.window_size, power)
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
    
        x = np.arange(window_size)
        def gaus(x, x0, k):
            std = np.sqrt( x0**2 / (8*k))
            a = 2 / (1 - np.exp(-k))
            c = 2 - a
            y = a*np.exp(-((x-(x0/2))**2)/(2*std**2)) + c
            
            return y
        window = gaus(x, len(x), k)
        
        return window
     
    def predict_view(self, rotation):
        
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
    
    def average_predicted_views(self):
        
        for rotation in self.rotations:
            
            current_view = self.predict_view(rotation)
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
            prediction = prediction * self.window
            z,y,x = pos_l[i]
            self.pred_img[z: z+window_size, y: y+window_size, x:x+window_size] = self.pred_img[z: z+window_size, y: y+window_size, x:x+window_size] + prediction 
        
        gc.collect()
        
        #empty queue
        self.batch_queue = []
        
        return self
        
    def _normalize(self, subdivisions = None):
        if subdivisions is None:
            subdivisions = self.subdivisions
        
        self.pred_img = self.pred_img / (subdivisions **2)
        
    def _back_transform(self):

        flipn, ax, k = self.rot_id
        #rot first, flip last
        self._unrot(ax,k)
        self._unflip(flipn)
        self._unpad()
        
    def _unflip(self, flipn):
        flip = self.flip_axes[flipn]
        self.pred_img = np.flip(self.pred_img,axis = flip)
        
    def _unrot(self, ax, k):
        self.pred_img = np.rot90(self.pred_img,k = -k, axes = self.rot_axes[ax])
        
    def _unpad(self):
        # aug = self.aug
        z_min, z_max = self.padding[0]
        y_min, y_max = self.padding[1]
        x_min, x_max = self.padding[2]
        
        self.pred_img = self.pred_img[z_min : -z_max, y_min : -y_max, x_min : -x_max, :]
        
    def plot_padded(self, idx = 0):
        plt.imshow(self.padded_img[idx, :, :,0])
        plt.colorbar()
        plt.show()
        
    def plot_pred(self, idx = 0):
        plt.imshow(self.pred_img[idx, :, :,0])
        plt.colorbar()
        plt.show()
        
    def predict_from_patches(self):

        padz_len, pady_len, padx_len = self.padded_img.shape[:-1]
        
        step = int(self.window_size/self.subdivisions)
        
        x_points = range(0, padx_len-self.window_size+1, step)
        y_points = range(0, pady_len-self.window_size+1, step)
        z_points = range(0, padz_len-self.window_size+1, step)
   
        
        for z in z_points:
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



