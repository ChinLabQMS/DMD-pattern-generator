from PIL import Image, ImageDraw, ImageFont

import math
import numpy as np

# DMD dimensions
DMD_ROWS = 1140
DMD_COLS = 912

class PatternPainter:
    def __init__(self, nrows, ncols) -> None:
        """
        PatternPainter class is used to generate coordinates of patterns on a rectangular grid
        --------------------
        Parameters:
        --------------------
        nrows: int
            Number of rows in the rectangular grid
        ncols: int
            Number of columns in the rectangular grid
        """
        self.nrows = nrows
        self.ncols = ncols

    def drawCircle(self, row_offset=0, col_offset=0, radius=50):
        """
        Draw a circle on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            Row offset of the center of the circle
        col_offset: int
            Column offset of the center of the circle
        radius: int
            Radius of the circle

        --------------------
        Returns:
        --------------------
        corr: int
            Coordinates of the points in the circle
        """
        # Find the center coordinates
        center_row, center_col = self.nrows // 2, self.ncols // 2
        row, col = center_row + row_offset, center_col + col_offset

        # Draw a filled circle with the given radius
        ans = np.array([(i, j) for i in range(max(0, int(row-radius)), min(int(row+radius+1), self.nrows))\
                for j in range(max(0, int(col-radius)), min(int(col+radius+1), self.ncols)) \
                if (i-row)**2 + (j-col)**2 <= radius**2])
        return ans
    
    def drawArrayOfCircle(self, row_spacing=50, col_spacing=50, row_offset=0, col_offset=0, nx=5, ny=5, radius=1):
        """
        Draw an array of circles on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_spacing: int
            Spacing between rows of circles
        col_spacing: int
            Spacing between columns of circles
        row_offset: int
            Row offset of the center of the first circle
        col_offset: int
            Column offset of the center of the first circle
        nx: int
            Number of circles in each row
        ny: int
            Number of circles in each column
        radius: int
            Radius of the circles

        --------------------
        Returns:
        --------------------
        corr: int
            Coordinates of the points in the arrays of circles
        """
        corr = [self.drawCircle(row_offset=i*row_spacing+row_offset, 
                                col_offset=j*col_spacing+col_offset, 
                                radius=radius) for i in range(nx) for j in range(ny)]
        return np.concatenate(corr, axis=0)

class DMDImage:
    def __init__(self) -> None:
        """
        DMDImage class is used to store the DMD image in a 2D array of 1s and 0s, where 1 
        represents a white pixel (on) and 0 represents a black pixel (off).

        --------------------
        Attributes:
        --------------------
        template: PIL Image object
            The template image in real space, which is the image that will be converted to DMD space
        dmdarray: PIL Image object
            The DMD image in DMD space, which is the image that will be displayed on the DMD
        rows: int
            Number of rows in the DMD image
        cols: int
            Number of columns in the DMD image
        real_rows: int
            Number of rows in the real space image
        real_cols: int
            Number of columns in the real space image
        """
        self.nrows = DMD_ROWS
        self.ncols = DMD_COLS

        self.real_nrows = math.ceil((self.nrows-1) / 2) + self.ncols
        self.real_ncols = self.ncols + (self.nrows-1) // 2

        self.template = np.full((self.real_nrows, self.real_ncols, 3), (255, 0, 0), dtype=np.uint8)
        self.dmdarray = np.full((self.nrows, self.ncols, 3), 0, dtype=np.uint8)

        row, col = np.meshgrid(np.arange(self.nrows), np.arange(self.ncols), indexing='ij')
        self.dmdrows, self.dmdcols = self.realSpace(row.flatten(), col.flatten())

    def realSpace(self, row, col):
        """
        Convert the given DMD space row and column to real space row and column
        --------------------
        Parameters:
        --------------------
        row: int | array-like
            Row in DMD space
        col: int | array-like
            Column in DMD space
        
        --------------------
        Returns:
        --------------------
        real_row: int
            Row in real space
        real_col: int
            Column in real space
        """
        return (np.ceil(row/2)).astype(int) + col, self.ncols - 1 + row//2 - col
    
    def setTemplate(self, color=1):
        """
        Set the template image in real space to a solid color
        --------------------
        Parameters:
        --------------------
        color: int
            1 for white (on), 0 for black (off)
        """
        # Paint all pixels within DMD space to white/black, default is white (on)
        self.template[self.dmdrows, self.dmdcols, :] = color * np.array([255, 255, 255])
        self.dmdarray[:] = color * 255
    
    def getTemplateImage(self, color=1):
        """
        Return a PIL Image object of the template image in real space with labels on the corners
        --------------------
        Parameters:
        --------------------
        color: int
            1 for white (on), 0 for black (off)
        
        --------------------
        Returns:
        --------------------
        template: PIL Image object
            The template image in real space
        """
        self.setTemplate(color=color)
        image = Image.fromarray(self.template, mode='RGB')        
        
        # Add labels on the corners
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 30)

        corner00 = self.realSpace(0, 0)[1] - 100, self.realSpace(0, 0)[0]
        corner10 = self.realSpace(self.nrows-1, 0)[1] - 150, self.realSpace(self.nrows-1, 0)[0] + 150
        corner11 = self.realSpace(self.nrows-1, self.ncols-1)[1] + 50, self.realSpace(self.nrows-1, self.ncols-1)[0] - 50

        draw.text(corner00, '(0, 0)', font=font, fill=0)
        draw.text(corner10, f'({self.nrows-1}, 0)', font=font, fill=0)
        draw.text(corner11, f'({self.nrows-1}, {self.ncols-1})', font=font, fill=0)
        return image

    def convertImageToDMDArray(self, image):
        """
        Convert the given real space image to a DMD image
        --------------------
        Parameters:
        --------------------
        image: PIL Image object
            The real space image to be converted to DMD space
        """
        assert image.size == (self.real_ncols, self.real_nrows)
        self.template[:, :, :] = np.asarray(image, dtype=np.uint8)
        self.convertTemplateToDMDArray()
    
    def convertTemplateToDMDArray(self):
        """
        Convert the template image in real space to a DMD image
        """
        # Loop through every column and row for the DMD image and assign it 
        # the corresponding pixel value from the real space image
        self.dmdarray[:, :, :] = self.template[self.dmdrows, self.dmdcols, :].reshape(self.nrows, self.ncols, 3)
    
    def saveDMDArray(self, filename):
        """
        Save the DMD image to a BMP file
        --------------------
        Parameters:
        --------------------
        filename: str
            Name of the BMP file to be saved
        """
        image = Image.fromarray(self.dmdarray, mode='RGB')
        image.save(filename)
        print('DMD pattern saved as', filename)
    
    def drawPattern(self, corr, color=1, reset=False):
        """
        Draw a pattern on the DMD image at the given coordinates
        --------------------
        Parameters:
        --------------------
        corr: int | array-like
            Coordinates of the points in the pattern
        color: int
            1 for white (on), 0 for black (off)
        reset: bool
            True to reset the real space template to the default template, False otherwise

        --------------------
        Returns:
        --------------------
        template: PIL Image object
            The template image in real space
        """
        # Reset the real space template
        if reset: self.setTemplate(color=1-color)
        
        # Update the pixels on DMD array in real space
        self.template[corr[:, 0], corr[:, 1]] = color * np.array([255, 255, 255])
        self.convertTemplateToDMDArray()

        return Image.fromarray(self.template, mode='RGB')