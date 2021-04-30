from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

def augmentationsGenerator(gen, augGeneratorArgs):
    # Initialize the image data generator with args provided    
    for images, masks in gen:
        # See https://github.com/aleju/imgaug/issues/499
        augmenter = iaa.Affine(**augGeneratorArgs)
        augmenter._mode_segmentation_maps = "symmetric"
        
        images_aug, segmaps_aug = augmenter(images=images*255, segmentation_maps=masks.astype(np.int32))

        yield images_aug/255.0, segmaps_aug