from PIL import Image, ImageDraw, ImageFont

from tkinter import Tk
from tkinter.filedialog import askopenfilename

import math
import os

from DMD_pattern_function import *

DMD_ROWS = 1140
DMD_COLS = 912

REAL_ROWS = math.ceil((DMD_ROWS-1) / 2) + DMD_COLS
REAL_COLS = DMD_COLS + (DMD_ROWS-1) // 2

# Convert DMD space (row, col) to real space
def realSpaceRow(row, col):
    return math.ceil(row / 2) + col

def realSpaceCol(row, col):
    return DMD_COLS - 1 + row//2 - col

class DMDImage:
    
    def __init__(self) -> None:
        self.template = None
    
    def generateTemplate(self, color=1):
        template = Image.new("RGB", size=(REAL_COLS, REAL_ROWS), color='#ff0000')

        for row in range(DMD_ROWS):
            for col in range(DMD_COLS):
                real_row, real_col = realSpaceRow(row, col), realSpaceCol(row, col)
                if color == 1:
                    template.putpixel((real_col, real_row), value=(255, 255, 255))
                elif color == 0:
                    template.putpixel((real_col, real_row), value=(0, 0, 0))
        
        # Add labels on the corners
        draw = ImageDraw.Draw(template)
        font = ImageFont.truetype("arial.ttf", 30)

        corner00 = realSpaceCol(0, 0) - 100, realSpaceRow(0, 0)
        corner10 = realSpaceCol(DMD_ROWS-1, 0) - 150, realSpaceRow(DMD_ROWS-1, 0) + 150
        corner11 = realSpaceCol(DMD_ROWS-1, DMD_COLS-1) + 50, realSpaceRow(DMD_ROWS-1, DMD_COLS-1) - 50

        draw.text(corner00, '(0, 0)', font=font, fill=0)
        draw.text(corner10, f'({DMD_ROWS-1}, 0)', font=font, fill=0)
        draw.text(corner11, f'({DMD_ROWS-1}, {DMD_COLS-1})', font=font, fill=0)

        filename = f'DMD_template_{DMD_ROWS}x{DMD_COLS}.bmp'
        template.save(filename)
        print(f'Template saved as {filename}')

        self.template = template
        return

    @classmethod
    def convertImageToDMDArray(cls, image):
        pixels = image.load()

        #Create a DMD image of the correct dimensions
        dmd_image = Image.new("1", (DMD_COLS, DMD_ROWS), color=1)  

        #Loop through every column and row for the DMD image and assign it 
        #the corresponding pixel value from the real space image
        for row in range(DMD_ROWS):
            for col in range(DMD_COLS):
                real_row, real_col = realSpaceRow(row, col), realSpaceCol(row, col)
                dmd_image.putpixel((col, row), value=pixels[real_col, real_row])
        return dmd_image

if __name__ == '__main__':

    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title='Select Modified BMP Image Template', filetypes=[('BMP Files', '*.bmp')])
    directory, filename = os.path.split(file_path)
    filename_new = 'DMD_pattern_' + os.path.splitext(filename)[0] + '.bmp'

    image = Image.open(file_path)
    image = image.convert('1')

    #Convert the template to a DMD Image
    dmd_image = DMDImage.convertImageToDMDArray(image)

    #Show the converted DMD pattern and save it to your directory
    dmd_image.show()
    dmd_image.save(os.path.join(directory, filename_new))