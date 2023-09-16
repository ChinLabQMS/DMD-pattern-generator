from PIL import Image, ImageDraw, ImageFont

from tkinter import Tk
from tkinter.filedialog import askopenfilename

import math
import os

# DMD dimensions
DMD_ROWS = 1140
DMD_COLS = 912

class DMDPatternPainter:
    def __init__(self, dmd):
        self.dmd = dmd

    def drawCircle(self, row_offset=0, col_offset=0, radius=50, color=1):
        # Find the real space center coordinates
        center_row, center_col = self.dmd.template.size[1]//2, self.dmd.template.size[0]//2
        row, col = center_row + row_offset, center_col + col_offset

        # Draw a circle with the given radius and color on the DMD template
        for i in range(max(0, row-radius), min(row+radius+1, self.dmd.template.size[1])):
            for j in range(max(0, col-radius), min(center_col+radius+1, self.dmd.template.size[0])):
                
                if (i-row)**2 + (j-col)**2 <= radius**2:
                    if color == 1:
                        self.dmd.template.putpixel((j, i), value=(255, 255, 255))
                    elif color == 0:
                        self.dmd.template.putpixel((j, i), value=(0, 0, 0))
        
        # Update the DMD array
        self.dmd.convertTemplateToDMDArray()

        return

class DMDImage: 
    def __init__(self) -> None:
        self.template = None
        self.dmdarray = Image.new("1", (DMD_COLS, DMD_ROWS), color=1)

        self.rows = DMD_ROWS
        self.cols = DMD_COLS

        self.real_rows = math.ceil((self.rows-1) / 2) + self.cols
        self.real_cols = self.cols + (self.rows-1) // 2

    # Convert DMD space (row, col) to real space
    def realSpaceRow(self, row, col):
        return math.ceil(row / 2) + col

    def realSpaceCol(self, row, col):
        return self.cols - 1 + row//2 - col
    
    def setTemplate(self, color=1):
        # Create a red image in real space
        template = Image.new("RGB", size=(self.real_cols, self.real_rows), color='#ff0000')

        # Paint all pixels within DMD space to white/black, default is white (on)
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
        self.convertTemplateToDMDArray()

        return
    
    def getTemplate(self, color=1):
        self.setTemplate(color=color)
        return self.template

    def convertImageToDMDArray(self, image):
        pixels = image.load()
        assert image.size == (self.real_cols, self.real_rows)
        
        # Loop through every column and row for the DMD image and assign it 
        # the corresponding pixel value from the real space image
        for row in range(self.rows):
            for col in range(self.cols):
                real_row, real_col = self.realSpaceRow(row, col), self.realSpaceCol(row, col)
                self.dmdarray.putpixel((col, row), value=pixels[real_col, real_row])
        return
    
    def convertTemplateToDMDArray(self):
        self.convertImageToDMDArray(self.template.convert('1'))
        return
    
    def showDMDArray(self):
        self.dmdarray.show()
        return
    
    def saveDMDArray(self, filename):
        self.dmdarray.save(filename)
        return

if __name__ == '__main__':

    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title='Select modified BMP Image template', filetypes=[('BMP Files', '*.bmp')])
    directory, filename = os.path.split(file_path)
    filename_new = 'DMD_pattern_' + os.path.splitext(filename)[0] + '.bmp'

    # Convert the loaded modified template to a DMD Image
    dmd_image = DMDImage()
    dmd_image.convertImageToDMDArray(Image.open(file_path).convert('1'))

    # Show the converted DMD pattern and save it to your directory
    dmd_image.showDMDArray()
    dmd_image.saveDMDArray(os.path.join(directory, filename_new))