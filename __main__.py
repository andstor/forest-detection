# %%
%load_ext autoreload
%autoreload 2

import tensorflow as tf

from keras_unet.models import custom_unet, vanilla_unet, satellite_unet

from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance

from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from preprocessing.filter_dataset import filterDataset
from visualization_generator import visualizeGenerator

from preprocessing.data_generator import dataGeneratorCoco
from preprocessing.data_augumentation import augmentationsGenerator
from preprocessing.patch_generator import patchGenerator

import imgaug.augmenters as iaa

import random


folder = './dataset'
classes = None

# TODO:create validation datasets
train_images, dataset_size_train, coco_train = filterDataset(folder, classes, 'train')
val_images, dataset_size_val, coco_val = filterDataset(folder, classes, 'train')

print("\n")
print("Training dataset size: ", dataset_size_train)
print("Validation dataset size: ", dataset_size_val)

batch_size = 1 # >1 results in warped images because of the data agumentor.
patch_size = 32 # Should be cubic
patch_batch_size = 4 #
steps_per_epoch_multiplyer = 2

input_image_size = (512, 512) # (Height, Width) - image to be turned into patches
channels = 4 # red, blue, green
input_shape = (patch_size, patch_size, channels) # Network input size (Height, Width, Channels)
mask_type = 'binary' # 'normal'


augGeneratorArgsold = dict(featurewise_center = False, 
                        samplewise_center = False,
                        rotation_range = 5, 
                        width_shift_range = 0.01, 
                        height_shift_range = 0.01, 
                        brightness_range = (0.8,1.2),
                        shear_range = 0.01,
                        zoom_range = [1, 1.25],  
                        horizontal_flip = True, 
                        vertical_flip = False,
                        fill_mode = 'reflect',
                        data_format = 'channels_last')

augGeneratorArgs = dict(
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-90, 90), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode="symmetric" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )

# Create training generators
train_gen = dataGeneratorCoco(train_images, classes, coco_train, folder,
                            input_image_size, channels, batch_size, 'train', mask_type)
train_gen_aug = augmentationsGenerator(train_gen, augGeneratorArgs)
train_gen_patch = patchGenerator(train_gen_aug, patch_size, patch_batch_size)

# Create validation generators
val_gen = dataGeneratorCoco(val_images, classes, coco_val, folder,
                            input_image_size, channels, batch_size, 'train', mask_type)
val_gen_patch = patchGenerator(val_gen, patch_size, patch_batch_size)


# Set  parameters
n_epochs = 1

n_patches = (input_image_size[0] // patch_size) * (input_image_size[1] // patch_size)
steps_per_epoch = ((dataset_size_train * (n_patches // patch_batch_size)) // batch_size) * steps_per_epoch_multiplyer
validation_steps = (dataset_size_val * (n_patches // patch_batch_size)) // batch_size
opt = SGD(lr=0.01, momentum=0.9) # Optimizer
#opt = Adam()
lossFn = 'binary_crossentropy' # Loss function
#lossFn = 'jaccard_distance'

print("Number of patches:", n_patches)
print("Steps per epoch (training): ", steps_per_epoch)
print("Steps per epoch (validation): ", validation_steps)
print("\n")

visualizeGenerator(train_gen_patch)

model = custom_unet(
    input_shape,
    use_batch_norm=True,
    num_classes=1,
    filters=64,
    num_layers=4,
    dropout=0.1,
    output_activation='sigmoid'
)

model.summary()
plot_model(model, to_file='model.png')


model_filename = 'segm_model_v0.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename, 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True,
)

# Compile model
model.compile(optimizer=opt, loss=lossFn, metrics=[iou, iou_thresholded])


#%% Start the training process
from keras_unet.utils import plot_segm_history

history = model.fit(x = train_gen_patch,
                validation_data = val_gen_patch,
                steps_per_epoch = steps_per_epoch,
                validation_steps = validation_steps,
                epochs = n_epochs,
                callbacks=[callback_checkpoint],
                verbose = True)


plot_segm_history(history)


#%% Test predict
from keras_unet.utils import plot_imgs

model.load_weights(model_filename)

x_val, y_val = next(train_gen_patch)

print("train shape:", x_val.shape)
print("mask shape:", y_val.shape)
pred = model.predict(x_val)
print("pred shape:", pred.shape)
x_val = x_val[:,:,:,:3]
plot_imgs(
    org_imgs=x_val, # required - original images
    mask_imgs=y_val, # required - ground truth masks
    pred_imgs=pred, # optional - predicted masks
    nm_img_to_plot=x_val.shape[0], # optional - number of images to plot
    color="red",
    fontsize=25
)


#%%

from imgaug import augmenters as iaa
import skimage.io as io
import matplotlib.pyplot as plt

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])
img = io.imread('test/images/T32VPQ_20200810T103629_TCI_10m_C0_R18.jpg')
print(img.shape)
rgb = np.dstack([img,img])
print(rgb.shape)

images_aug1 = seq(images=rgb)
images_aug2 = seq(images=rgb)
plt.figure()
plt.imshow(images_aug1[:,:,1])
plt.figure()
plt.imshow(images_aug2[:,:,1:4])

#%% Print prediction examples from reconstructed patches

from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import cv2

TCI = io.imread('test/images/T32VPQ_20200810T103629_TCI_10m_C0_R18.jpg')/255.0
B08 = io.imread('test/images/T32VPQ_20200810T103629_B08_10m_C0_R18.jpg', as_gray=True)/255.0
image = np.dstack([TCI, B08])

# Resize
image = cv2.resize(image, tuple(reversed(input_image_size)))
TCI = cv2.resize(TCI, tuple(reversed(input_image_size)))

print("img shape", image.shape)
dim = 32
patches = patchify(image, (dim,dim, image.shape[-1]), step=dim) # split image into 4*4 small 128*128 patches.
print("patches",patches.shape)

"""

plt.figure(figsize=(10, 10))
for imgs in patches:
    count = 0
    for r in range(16):
        for c in range(16):
            ax = plt.subplot(16, 16, count+1)
            plt.imshow(patches[r][c][0][:,:,:3])
            count += 1
            ax.axis('off')
reconstructed_image = unpatchify(patches, image.shape)
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(reconstructed_image[:,:,:3])
"""

patches = patches.reshape(-1, *patches.shape[-3:])
print("reshaped", patches.shape)
print("patches",patches.shape)

from keras_unet.utils import plot_imgs

model.load_weights(model_filename)
#model.load_weights("segm_model_v0_copy.h5")

pred = model.predict(patches)

"""
x_val = patches
y_val = patches
print("train shape:", x_val.shape)
print("mask shape:", y_val.shape)
pred = model.predict(x_val)
print("pred shape:", pred.shape)
plot_imgs(
    org_imgs=x_val[:,:,:,:3], # required - original images
    mask_imgs=y_val[:,:,:,:3], # required - ground truth masks
    pred_imgs=pred[:,:,:,:], # optional - predicted masks
    nm_img_to_plot=10, # optional - number of images to plot
    alpha=1,
    color="red"
    )
"""


print(pred.shape)
print(TCI.shape)
patches2 = pred.reshape(16, 16, 1, *pred.shape[-3:])
print("reconstruct", patches2.shape)

reconstructed_image = unpatchify(patches2, (512,  512, 1))
masked_data = reconstructed_image
#masked_data = np.where((masked_data)>0.85, 1, 0)
masked_data = np.ma.masked_where(masked_data < 0.5, masked_data)
#masked_data = np.ma.masked_where(masked_data < 0.75, masked_data)
#masked_data = np.ma.masked_where(masked_data <= 0, masked_data)

plt.figure(figsize=(10, 10))
plt.imshow(TCI, 'jet', interpolation='none')
plt.savefig('foo.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 10))
#plt.imshow(TCI, cmap='jet', interpolation='none')
im = plt.imshow(masked_data, cmap='gray', interpolation='none', alpha=1)
#plt.colorbar(im)
#plt.axis('off')
plt.savefig('bar.png', bbox_inches='tight')
plt.show()

"""
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1,2,1)
plt.imshow(image, 'jet', interpolation='none')
plt.subplot(1,2,2)
plt.imshow(image, 'jet', interpolation='none')
#plt.imshow(masked_data, 'gray', interpolation='none', alpha=0.9)
im = plt.imshow(masked_data, interpolation='none', alpha=0.9)
plt.colorbar(im);
plt.show()
"""
