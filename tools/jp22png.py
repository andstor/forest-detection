import sys
import os
import numpy as np
import cv2

class JP22JPG:
    def __init__(self, imgPath):
        self.imgPath = imgPath
        base = os.path.splitext(self.imgPath)[0]
        self.outPath = base + '.png'
        
        
    def convert(self):
        img = cv2.imread(self.imgPath)
        cv2.imwrite(self.outPath, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == "__main__":
    filePath = None
    grid = (3,3)
    if len(sys.argv) == 2:
        filePath = sys.argv[1]
    else:
        print("Invalid arguments!")
        pass

    dataFrag = JP22JPG(filePath)
    dataFrag.convert()
    print("Conversion complete")