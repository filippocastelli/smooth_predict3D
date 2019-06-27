# 3D Smoothly Blended Patch-Based Prediction of Extended Volumes

# <img src="/img/sp3d.png" alt="Drawing" width = "400"></img>
Performs 3D patch-based, prediction of extended volumes averaging over different
flips and rotations of the same volume to reduce noise.

For every view each voxel is predicted n times using sliding 3D windows, border
effects are minimized using a 3D second order spline window function to prioritize
the central area of the receptive field.

Implementation details to be added soon.

Derivative work of 2D analogous project https://github.com/Vooban/Smoothly-Blend-Image-Patches by Vooban Inc
# How to use:

```python
from smooth_predict3D import smooth_predict3D

from your_code import your_model

# Instanciate a U-Net CNN (or any similarly-behaved neural network) in the variable named `model`. We use a Keras model but it can be anything:
model = your_model()

# CNN's receptive field's border size: size of patches
window_size = 160

# Amount of categories predicted per pixels.
nb_classes = 10

# Load an image. Convention is channel_last, such as having an input_img.shape of: (x, y, z, nb_channels), where nb_channels is of 3 for regular RGB images. If your model has a different input scheme pre-transform the data.
input_img = ...

# Use the algorithm, 'pred_func' is used to predict batches of patches, in (batch, x, y, z, nb_channels), if your model has a different input scheme define pred_func as a function to accept (batch, x,y,z, channels) as input and return (batch, x,y,z, predict_classes)

# Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, z, nb_channels), such as a Keras model.

predictions_smooth = predict_img_with_smooth_windowing(
    input_img,
    window_size=window_size,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    nb_classes=nb_classes,
    pred_func=(
        lambda img_batch_subdiv: model.predict(image_to_neural_input(img_batch_subdiv))
    ),
    max_batch = 10 # Maximum numbers of patches in a batch
)

# For more details, refer to comments in code
```
Coded by [Filippo Castelli](https://github.com/filippocastelli)

Forked from original project by [Guillaume Chevalier](https://github.com/guillaume-chevalier)

[MIT License](https://github.com/filippocastelli/smooth_predict3D/blob/master/LICENSE). Copyright (c) 2019 Filippo Maria Castelli

## For more information

If you need anything don't hesitate to open an issue on the repo page!
