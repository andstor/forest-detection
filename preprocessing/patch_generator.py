from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
import numpy as np
import math
import random

def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

def patchGenerator(gen, patch_size=128, patch_batch_size=1):
    """Image generator splitting image into smaller patches."""
    
    for imgs, masks in gen: # For each batch
        img_list = []
        mask_list = []
        for i in range(0, imgs.shape[0]): # For each image in a batch
            patch_x = patchify(imgs[i], (patch_size, patch_size, imgs[i].shape[-1]), step=patch_size) # split image into 4*4 small 128*128 patches.
            img_p = patch_x.reshape(-1, *patch_x.shape[-3:])
            img_list.append(img_p)

            mask_y = patchify(masks[i], (patch_size, patch_size, 1), step=patch_size) # split mask into 4*4 small 128*128 patches.
            mask_p = mask_y.reshape(-1, *mask_y.shape[-3:])
            mask_list.append(mask_p)
            
            if (patch_batch_size == 1):
                for j in range(0, img_p.shape[0]): # For each patch in a image
                    yield img_p[j][np.newaxis, :], mask_p[j][np.newaxis, :]
        
        if (patch_batch_size > 1):
            image_patches = np.concatenate(img_list)
            mask_patches = np.concatenate(mask_list)
            patch_batch_counter = 0
            for idx in range(0, patch_batch_size):
                image_patch_batch = image_patches[patch_batch_counter:patch_batch_counter + patch_batch_size]
                mask_patch_batch = mask_patches[patch_batch_counter:patch_batch_counter + patch_batch_size]
                shuffled_images, shuffled_masks = randomize(image_patch_batch, mask_patch_batch)
                yield shuffled_images, shuffled_masks