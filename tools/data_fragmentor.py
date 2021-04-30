import sys
import os
import numpy as np
from PIL import Image

class DataFragmentor:
    def __init__(self, imgPath, grid, outPath="croppedImage"):
        self.imgPath = imgPath
        self.img = Image.open(imgPath)
        self.gridShape = (int(grid[0]), int(grid[1]))
        self.outPath = outPath
    
    def sliceImg(self):
        imgName = self.imgPath.split(".")[0]
        if not os.path.exists(imgName):
            os.makedirs(imgName)
        imgWidth, imgHeight = self.img.size
        width = int(imgWidth / self.gridShape[0])
        height = int(imgHeight / self.gridShape[1])
        for i in range(0,self.gridShape[0]):
            r = i * width
            for j in range(0,self.gridShape[1]):
                c = j * height
                box = (r, c, r+width, c+height)
                cropped = self.img.crop(box)
                try:
                    outputPath = "{}/C{}_R{}.jpg".format(imgName, i, j)
                    cropped.save(outputPath)
                except:
                    pass


if __name__ == "__main__":
    filePath = "test.jpg"
    grid = (4,4)
    if len(sys.argv) == 2:
        filePath = sys.argv[1]
    elif len(sys.argv) == 4:
        filePath = sys.argv[1]
        grid = (sys.argv[2], sys.argv[3])
    else:
        print("Invalid arguments!")
        pass

    dataFrag = DataFragmentor(filePath, grid)
    dataFrag.sliceImg()
    print("Slicing complete")