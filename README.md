# ResMem

This is a package that wraps [ResMem](https://coen.needell.co/project/memnet/). This is a residual neural network that 
estimates the memorability of an image, on a scale of 0 to 1.

## How to Use
To install from PyPI:
```shell
pip install resmem
```

The package contains two objects, ResMem itself, and a preprocessing transformer function built on torchvision.
```python
from resmem import ResMem, transformer

model = ResMem(pretrained=True)

``` The `transformer` is designed to be used with pillow.

```python
from PIL import Image

img = Image.open('./path/to/image') # This loads your image into memory
img = img.convert('RGB') 
# This will convert your image into RGB, for instance if it's a PNG (RGBA) or if it's black and white.

model.eval()
# Set the model to inference mode.

image_x = transformer(img)
# Run the preprocessing function

prediction = model(image_x.view(-1, 3, 227, 227))
# For a single image, the image must be reshaped into a batch
# with size 1.
# Get your prediction!
```

For more advanced usage, see the `sample.py` file in this repository.

## Description of the Model

Historically, the next big advancement in using neural networks after AlexNet, the basis for MemNet, was ResNet. This allowed for convolutional neural networks to be built deeper, with more layers, without the gradient exploding or vanishing. Knowing that, let's try to include this in our model. What we will do is take a pre-trained ResNet, that is the whole thing, not just the convolutional features, and add it as an input feature for our regression step. The code for this is [here.](https://www.coeneedell.com/appendix/memnet_extras/#resmem)

![ResMem Diagram](ResMem.jpg)

For the following discussion, while ResMem is initialized to a pre-trained optimum, I have allowed it to retrain for our problem. The thought is that given a larger set of parameters the final model *should* be more generalizable. Using weights and biases, we can run a hyperparameter sweep.

![ResMem Testing](resnetsweep.png)

Here we can see much higher peaks, reaching into the range of 0.66-0.67! All of these runs were both trained and validated on a dataset that was constructed from both MemCat and LaMem databases.

## Github Instructions

If you want to fiddle around with the raw github sourcecode, go right ahead. But be aware that we use git lfs for the 
model storage. You'll have to install the package at https://git-lfs.github.com/ and then run:
```shell
git lfs install
git lfs pull
```
to get access to the full model. Also note that the `tests.py` file that's included with this repository references a 
database that we, at this time, do not have permission to distribute, so consider that test file to be a guide rather 
than a hard-and-fast test.

If you cannot access the model through git lfs, you can get a fresh copy of the model file from the pypi version of the package, or you can download the weights from (osf)[https://osf.io/qf5ry/].

## Analysis

New in 2023:

ResMem now includes an analysis module. This is a set of tools for accessing the direct activations of specific filters within the convolutional neural network.
Generally features can be accessed by:

1. Selecting a filter to record data from.
2. Running an image through the model.
3. Recording the activation of the selected filter.

This behavior is implemented using the `SaveFeatures` class in the `analysis` module.
For example:

```python
from resmem.model import ResMem, transformer
from resmem.analysis import SaveFeatures
import numpy as np
from PIL import Image

# Generating a random image for demonstration purposes. You would want to use the image you're analyzing.
randim = np.random.randint(0, 256, (256, 256, 3))
randim = Image.fromarray(randim, mode='RGB')

# Set up ResMem for prediction
model = ResMem(pretrained=True)
model.eval()

activation = SaveFeatures(list(list(model.features.children())[5].children())[4])
# Notice the numbers here, we have to select specific features a priori. We can't just record everything, because it would quickly fill up the computer's RAM.
# In the SaveFeatures object, we are selecting a specific *Residual Block*. The blocks are organized into "sections". Above we select section 5, block 4. Note that the children selector can also select non-convolutional modules within ResMem
memorability = model(transformer(randim).view(-1, 3, 227, 227))
# Now the activation object contains some data.
# The first index here should always be zero, it's selecting a batch index. In the case where you're running this in batches, you'll want to slice across that index instead.
# The second index selects a filter in that layer (the filter associated with a certain channel).
activation.features[0, 1]
# Even then, this filter activation is a 29x29 tensor, one for each time the filter was applied to some subimage.
# For analyzing filter activations, you may simply average these individual neuron-level activations.
activation.features[0, 1].mean()
```

This is all pretty complicated, and requires referencing a map of the submodules within ResMem. 
For example, "sections" 0-3 are all single-module transformations, so their activation information is pretty much useless, and there are no children to descend into.
"Section" 4 has three residual blocks, each of which has three convolutional layers, three batch norms, a relu, and a resampling filter.
The point here is that sections are not of consistent size.
So, we've implemented a few helper structures for common tasks.

### Getting convolutional layers by depth

Despite being named "ResNet-152," there are 155 convolutional layers in ResMem's resnet branch.
Using the new interface, we can examine a specific depth, ignoring non-convolutional modules in the model.

```python
...

activation = SaveFeatures(model.resnet_layers[24])
memorability = model(transformer(randim).view(-1, 3, 227, 227))
activation.features[0, 1].mean()
# or to get all features from that layer
activation.features[0, :].mean(dim=(1,2))
```

We wrap all of this in the new `ResMem.resnet_activation()` method.

```python
activations = model.resnet_activation(transformer(randim).view(-1, 3, 227, 227), 24)
```

This method does support batching, and will return the activation of every filter in the selected layer.



## Citation

```
@article{Needell_Bainbridge_2022,
title={Embracing New Techniques in Deep Learning for Estimating Image Memorability},
volume={5},
ISSN={2522-0861,
2522-087X},
url={https://link.springer.com/10.1007/s42113-022-00126-5},
DOI={10.1007/s42113-022-00126-5},
number={2},
journal={Computational Brain & Behavior},
author={Needell, Coen D. and Bainbridge, Wilma A.},
year={2022},
month=jun,
pages={168–184},
language={en} }
```
