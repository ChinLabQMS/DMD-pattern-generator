from PIL import Image, ImageDraw, ImageFont

from tkinter import Tk
from tkinter.filedialog import askopenfilename

import math
import os

# DMD dimensions
DMD_ROWS = 1140
DMD_COLS = 912

class DMDPattern:
    def __init__(self) -> None:
        self.image = DMDImage()

    def circle_bright(template, col, row, radius=50):
        assert isinstance(template, Image.Image)

        for i in range(max(0, row-radius), min(row+radius+1, template.size[1])):
            for j in range(max(0, col-radius), min(col+radius+1, template.size[0])):

                if (i-row)**2 + (j-col)**2 <= radius**2:
                    template.putpixel((i, j), value=1)
        
        return template

class DMDImage: 
    def __init__(self) -> None:
        self.template = None
        self.rows = DMD_ROWS
        self.cols = DMD_COLS

        self.real_rols = math.ceil((self.rows-1) / 2) + self.cols
        self.real_cols = self.cols + (self.rows-1) // 2

    # Convert DMD space (row, col) to real space
    def realSpaceRow(self, row, col):
        return math.ceil(row / 2) + col

    def realSpaceCol(self, row, col):
        return self.cols - 1 + row//2 - col
    
    def generateTemplate(self, color=1):
        # Create a red image in real space
        template = Image.new("RGB", size=(self.real_cols, self.real_rols), color='#ff0000')

        # Paint all pixels within DMD space to white/black
        for row in range(self.rows):
            for col in range(self.cols):
                real_row, real_col = self.realSpaceRow(row, col), self.realSpaceCol(row, col)
                if color == 1:
                    template.putpixel((real_col, real_row), value=(255, 255, 255))
                elif color == 0:
                    template.putpixel((real_col, real_row), value=(0, 0, 0))
        
        # Add labels on the corners
        draw = ImageDraw.Draw(template)
        font = ImageFont.truetype("arial.ttf", 30)

        corner00 = self.realSpaceCol(0, 0) - 100, self.realSpaceRow(0, 0)
        corner10 = self.realSpaceCol(self.rows-1, 0) - 150, self.realSpaceRow(self.rows-1, 0) + 150
        corner11 = self.realSpaceCol(self.rows-1, self.cols-1) + 50, self.realSpaceRow(self.rows-1, self.cols-1) - 50

        draw.text(corner00, '(0, 0)', font=font, fill=0)
        draw.text(corner10, f'({self.rows-1}, 0)', font=font, fill=0)
        draw.text(corner11, f'({self.rows-1}, {self.cols-1})', font=font, fill=0)

        self.template = template
        return
    
    def getTemplate(self, color=1):
        if self.template is None:
            self.generateTemplate(color=color)
        return self.template

    def convertImageToDMDArray(self, image):
        pixels = image.load()
        assert image.size == self.real_cols, self.real_rols
        
        # Create a DMD image of the correct dimensions
        dmd_image = Image.new("1", (self.cols, self.rows), color=1)  

        # Loop through every column and row for the DMD image and assign it 
        # the corresponding pixel value from the real space image
        for row in range(self.rows):
            for col in range(self.cols):
                real_row, real_col = self.realSpaceRow(row, col), self.realSpaceCol(row, col)
                dmd_image.putpixel((col, row), value=pixels[real_col, real_row])
        return dmd_image

if __name__ == '__main__':

    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title='Select modified BMP Image template', filetypes=[('BMP Files', '*.bmp')])
    directory, filename = os.path.split(file_path)
    filename_new = 'DMD_pattern_' + os.path.splitext(filename)[0] + '.bmp'

    # Open the image and convert to binary format
    image = Image.open(file_path).convert('1')

    # Convert the template to a DMD Image
    dmd_image = DMDImage().convertImageToDMDArray(image)

    # Show the converted DMD pattern and save it to your directory
    dmd_image.show()
    dmd_image.save(os.path.join(directory, filename_new))