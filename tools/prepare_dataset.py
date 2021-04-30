
# %%
# Rename images and files
import os 
import json

images = "dataset/images"
annotations = "dataset/annotations"

#Rename images assuming PNG
for i, img_dir in enumerate(sorted(os.listdir(images))):
        image_p = os.path.join(images, img_dir)
        for img in sorted(os.listdir(image_p)):
            image_name = os.path.join(image_p, img)
            os.rename(image_name, image_name[:-4] + "_d" + str(i+1) + image_name[-4:])

# Rename images in annotation file  
for i, js in enumerate(sorted(os.listdir(annotations))):
    json_p = os.path.join(annotations, js)
    with open(json_p) as j:
        data = json.load(j)
        # Access image dict
        for im in data['images']:
            # Change image name
            im['file_name'] = im['file_name'][:-4] + "_d" + str(i+1) + im['file_name'][-4:]
    with open(json_p[:-5] + "_modified" + ".json", 'w') as outfile:
        json.dump(data, outfile)
# %%




# %%
from coco_assistant import COCO_Assistant

# Specify image and annotation directories
img_dir = os.path.join(os.getcwd(), 'dataset', 'images')
ann_dir = os.path.join(os.getcwd(), 'dataset', 'annotations')

print(img_dir)
# Create COCO_Assistant object
cas = COCO_Assistant(img_dir, ann_dir)
# %%
cas.merge(merge_images=True)

# %%
cas.visualise()
# %%
