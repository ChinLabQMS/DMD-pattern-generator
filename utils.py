from PIL import Image, ImageDraw, ImageFont

import math
import numpy as np
import os

# DMD dimensions
DMD_ROWS = 1140
DMD_COLS = 912

# Whether to filp the image vertically
FLIP = True

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

    def drawCircle(self, 
                   row_offset=0, 
                   col_offset=0, 
                   radius=50):
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
        corr: array-like of shape (N, 2)
            Coordinates of the points in the circle
        """
        # Find the center coordinates
        center_row, center_col = self.nrows // 2, self.ncols // 2
        row, col = center_row + row_offset, center_col + col_offset

        # Draw a filled circle with the given radius
        ans = [(i, j) for i in range(max(0, int(row-radius)), min(int(row+radius+1), self.nrows))\
                for j in range(max(0, int(col-radius)), min(int(col+radius+1), self.ncols)) \
                if (i-row)**2 + (j-col)**2 <= radius**2]
        return np.array(ans)
    
    def drawArrayOfCircles(self, 
                          row_spacing=50, 
                          col_spacing=50, 
                          row_offset=0, 
                          col_offset=0, 
                          nx=5, 
                          ny=5, 
                          radius=1):
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
        corr: array-like of shape (N, 2)
            Coordinates of the points in the arrays of circles
        """
        if isinstance(nx, int):
            assert nx >= 0, 'nx must be a non-negative integer'
            nx = list(range(nx))
        if isinstance(ny, int):
            assert ny >= 0, 'ny must be a non-negative integer'
            ny = list(range(ny))
        corr = []
        for i in nx:
            for j in ny:
                new_circle = self.drawCircle(row_offset=i*row_spacing+row_offset, 
                                        col_offset=j*col_spacing+col_offset, 
                                        radius=radius)
                if new_circle.shape[0] != 0:
                    corr.append(new_circle)
        return np.concatenate(corr, axis=0)
    
    def drawHorizontalLine(self, row_offset=0, line_width=1):
        """
        Draw a horizontal line on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            Row offset of the center of the line
        col_offset: int
            Column offset of the center of the line
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the line
        """
        # Find the center coordinates
        row = self.nrows // 2 + row_offset

        assert row >= 0 and row < self.nrows, 'Row offset out of range'
        ans = np.array([(row + i, j) for j in range(self.ncols) for i in range(line_width)])
        return ans
    
    def drawVerticalLine(self, col_offset=0, line_width=1):
        """
        Draw a vertical line on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            Row offset of the center of the line
        col_offset: int
            Column offset of the center of the line
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the line
        """
        # Find the center coordinates
        col = self.ncols // 2 + col_offset

        assert col >= 0 and col < self.ncols, 'Column offset out of range'
        ans = np.array([(i, col + j) for i in range(self.nrows) for j in range(line_width)])
        return ans
    
    def drawCross(self, row_offset=0, col_offset=0, line_width=1):
        """
        Draw a cross on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            Row offset of the center of the cross
        col_offset: int
            Column offset of the center of the cross

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the cross
        """
        return np.concatenate((self.drawHorizontalLine(row_offset=row_offset, line_width=line_width),
                               self.drawVerticalLine(col_offset=col_offset, line_width=line_width)), axis=0)
    
    def drawHorizontalLines(self, row_spacing=50, row_offset=0, line_width=1, ny=5):
        """
        Draw an array of horizontal lines on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_spacing: int
            Spacing between rows of lines
        row_offset: int
            Row offset of the center of the first line
        line_width: int
            Width of the lines
        ny: int | array-like
            Number of lines, or a list of row indices to draw lines on
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of lines
        """
        if isinstance(ny, int):
            assert ny >= 0, 'ny must be a non-negative integer'
            ny = list(range(ny))
        corr = [self.drawHorizontalLine(row_offset=i*row_spacing+row_offset, 
                                        line_width=line_width) for i in ny]
        return np.concatenate(corr, axis=0)
    
    def drawVerticalLines(self, col_spacing=50, col_offset=0, line_width=1, nx=5):
        """
        Draw an array of vertical lines on the rectangular grid
        --------------------
        Parameters:
        --------------------
        col_spacing: int
            Spacing between columns of lines
        col_offset: int
            Column offset of the center of the first line
                  
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of lines
        """
        if isinstance(nx, int):
            assert nx >= 0, 'nx must be a non-negative integer'
            nx = list(range(nx))
        corr = [self.drawVerticalLine(col_offset=j*col_spacing+col_offset, 
                                      line_width=line_width) for j in nx]
        return np.concatenate(corr, axis=0)
    
    def drawCrosses(self, row_spacing=50, col_spacing=50, row_offset=0, col_offset=0, line_width=1, nx=5, ny=5):
        """
        Draw an array of crosses on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_spacing: int
            Spacing between rows of crosses
        col_spacing: int
            Spacing between columns of crosses
        row_offset: int
            Row offset of the center of the first cross
        col_offset: int
            Column offset of the center of the first cross
        line_width: int
            Width of the crosses
        nx: int
            Number of crosses in each row
        ny: int
            Number of crosses in each column
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of crosses
        """
        if isinstance(nx, int):
            assert nx >= 0, 'nx must be a non-negative integer'
            nx = list(range(nx))
        if isinstance(ny, int):
            assert ny >= 0, 'ny must be a non-negative integer'
            ny = list(range(ny))
        corr = [self.drawCross(row_offset=i*row_spacing+row_offset, 
                               col_offset=j*col_spacing+col_offset, 
                               line_width=line_width) for i in nx for j in ny]
        return np.concatenate(corr, axis=0)
    
    def drawStar(self, row_offset=0, col_offset=0, num=10):
        """
        Draw a star on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            Row offset of the center of the star
        col_offset: int
            Column offset of the center of the star
        num: int
            Number of different sectors in the star

        --------------------
        Returns:
        --------------------
        corr: int
            Coordinates of the points in the star
        """
        # Find the center coordinates
        center_row, center_col = self.nrows // 2, self.ncols // 2
        row, col = center_row + row_offset, center_col + col_offset

        # Draw a star with the given number of sectors
        angle = 2 * np.pi / num
        rows, cols = np.meshgrid(np.arange(self.nrows), np.arange(self.ncols), indexing='ij')
        mask = ((np.arctan2(cols.flatten() - center_col, rows.flatten() - center_row) // angle) % 2).astype(bool)
        
        return np.stack((rows.flatten()[mask], cols.flatten()[mask])).transpose()
    
    def drawCheckerBoard(self, size=20):
        """
        Draw a checker board on the rectangular grid

        --------------------
        Parameters:
        --------------------
        size: int
            Side length of one checker board square
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the checker board
        """
        rows, cols = np.meshgrid(np.arange(self.nrows), np.arange(self.ncols), indexing='ij')
        mask = ((rows.flatten() // size) % 2 + (cols.flatten() // size) % 2) % 2
        return np.stack((rows.flatten()[mask.astype(bool)], cols.flatten()[mask.astype(bool)])).transpose()

class DMDImage:
    def __init__(self, flip=FLIP) -> None:
        """
        DMDImage class is used to store the DMD image in a 2D array of 1s and 0s, where 1 
        represents a white pixel (on) and 0 represents a black pixel (off).

        --------------------
        Parameters:
        --------------------
        flip: bool
            True to flip the image vertically, False otherwise.

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
        self.flip = flip

        self.real_nrows = math.ceil((self.nrows-1) / 2) + self.ncols
        self.real_ncols = self.ncols + (self.nrows-1) // 2

        # Initialize the template image in real space to red and the DMD image in DMD space
        self.template = np.full((self.real_nrows, self.real_ncols, 3), (255, 0, 0), dtype=np.uint8)
        self.dmdarray = np.full((self.nrows, self.ncols, 3), 0, dtype=np.uint8)

        row, col = np.meshgrid(np.arange(self.nrows), np.arange(self.ncols), indexing='ij')
        self.dmdrows, self.dmdcols = self.realSpace(row.flatten(), col.flatten())

        mask = np.full((self.real_nrows, self.real_ncols), True, dtype=bool)
        mask[self.dmdrows, self.dmdcols] = False

        real_row, real_col = np.meshgrid(np.arange(self.real_nrows), np.arange(self.real_ncols), indexing='ij')
        self.bgrows, self.bgcols = real_row[mask], real_col[mask]
       
        if self.bgrows.shape[0] + self.dmdrows.shape[0] != self.real_nrows * self.real_ncols:
            raise ValueError('Number of pixels in the DMD image does not match the number of pixels in the real space image')

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
        flip: bool
            True to flip the image vertically, False otherwise
        
        --------------------
        Returns:
        --------------------
        real_row: int | array-like
            Row in real space
        real_col: int | array-like
            Column in real space
        """        
        if self.flip: 
            real_row, real_col = (np.ceil((self.nrows - 1 - row)/2)).astype(int) + col, self.ncols - 1 + (self.nrows - 1 - row)//2 - col
        else:
            real_row, real_col = (np.ceil(row/2)).astype(int) + col, self.ncols - 1 + row//2 - col

        return real_row, real_col
    
    def setTemplate(self, color=1):
        """
        Set the template image in real space to a solid color
        --------------------
        Parameters:
        --------------------
        color: float | array-like
            1 for white (on), 0 for black (off), float for grayscale, list or array-like of shape (3,) for RGB
        """
        if isinstance(color, float) and color >= 0 and color <= 1 or \
            (isinstance(color, int) and color == 0 or color == 1):
            color = np.floor(255 * np.array([color, color, color])).astype(np.uint8)
        elif isinstance(color, list) and len(color) == 3 or \
            (isinstance(color, np.ndarray) and color.shape == (3,)) and np.all(color >= 0) and np.all(color <= 255):
            color = np.array(color).astype(np.uint8)
        
        # Initialize the template image in real space to red and the DMD image in DMD space
        self.template = np.full((self.real_nrows, self.real_ncols, 3), (255, 0, 0), dtype=np.uint8)

        # Paint all pixels within DMD space to white/black, default is white (on)
        self.template[self.dmdrows, self.dmdcols, :] = color
        self.dmdarray[:] = color
    
    def getTemplateImage(self):
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
        image = Image.fromarray(self.template, mode='RGB')
        
        # Add labels on the corners
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 30)

        if self.flip:
            offset = ((150, -150), (0, 50), (150, 0))
        else:
            offset = ((0, -100), (150, -150), (-50, 50))

        corner00 = self.realSpace(0, 0)[1] + offset[0][1], self.realSpace(0, 0)[0] + offset[0][0]
        corner10 = self.realSpace(self.nrows-1, 0)[1] + offset[1][1], self.realSpace(self.nrows-1, 0)[0] + offset[1][0]
        corner11 = self.realSpace(self.nrows-1, self.ncols-1)[1] + offset[2][1], self.realSpace(self.nrows-1, self.ncols-1)[0] + offset[2][0]

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
        assert image.size == (self.real_ncols, self.real_nrows), 'Image size does not match DMD template size'
        self.template[:, :, :] = np.asarray(image, dtype=np.uint8)
        self.convertTemplateToDMDArray()
    
    def convertTemplateToDMDArray(self):
        """
        Convert the template image in real space to a DMD image
        """
        # Loop through every column and row for the DMD image and assign it 
        # the corresponding pixel value from the real space image
        self.dmdarray[:, :, :] = self.template[self.dmdrows, self.dmdcols, :].reshape(self.nrows, self.ncols, 3)
    
    def saveDMDArray(self, dir, filename):
        """
        Save the DMD image to a BMP file
        --------------------
        Parameters:
        --------------------
        filename: str
            Name of the BMP file to be saved
        """
        if os.path.exists(dir) == False:
            os.makedirs(dir)
        dmd_filename = dir + 'pattern_' + filename
        template_filename = dir + 'template_' + filename

        image = Image.fromarray(self.dmdarray, mode='RGB')
        image.save(dmd_filename, mode='RGB')
        print('DMD pattern saved as', dmd_filename)

        image = self.getTemplateImage()
        image.save(template_filename, mode='RGB')
        print('Template image saved as', template_filename)
    
    def drawPattern(self, corr, color=1, reset=True):
        """
        Draw a pattern on the DMD image at the given coordinates
        --------------------
        Parameters:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the pattern
        color: int, color of the pattern
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
        self.template[self.bgrows, self.bgcols] = np.array([255, 0, 0])
        self.convertTemplateToDMDArray()