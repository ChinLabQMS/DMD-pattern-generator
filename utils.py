from PIL import Image, ImageDraw, ImageFont
from itertools import product
import math
import numpy as np
import os
from numba import jit
from matplotlib import pyplot as plt

# DMD dimensions
DMD_NROWS = 1140
DMD_NCOLS = 912

# 45 deg Real-space dimensions
REAL_NROWS = math.ceil((DMD_NROWS-1) / 2) + DMD_NCOLS
REAL_NCOLS = DMD_NCOLS + (DMD_NROWS-1) // 2

# Whether to filp the image vertically
FLIP = True
    
class Dither(object):
    @staticmethod
    def normalizePattern(image: np.ndarray):
        """
        Normalize the stored pattern to [0, 1], works in-place
        """
        max_val = image.max()
        min_val = image.min()
        if max_val == min_val:
            image.fill(0)
        else:
            np.copyto(image, (image - min_val) / (max_val - min_val))

    @staticmethod
    @jit(nopython=True)
    def floyd_steinberg(image: np.ndarray, inplace=True):
        """
        Floyd-Steinberg dithering algorithm.
        https://en.wikipedia.org/wiki/Floyd–Steinberg_dithering
        https://gist.github.com/bzamecnik/33e10b13aae34358c16d1b6c69e89b01
        --------------------
        Parameters:
        --------------------
        image: np.array of shape (height, width), dtype=float, 0.0-1.0
            works in-place!
        --------------------
        Returns:
        --------------------
        image: np.array of shape (height, width), dtype=float, 0.0-1.0
        """
        if not inplace: image = image.copy()
        h, w = image.shape        
        for y in range(h):
            for x in range(w):
                old = image[y, x]
                new = np.round(old)
                image[y, x] = new
                error = old - new

                if x + 1 < w:
                    image[y, x + 1] += error * 0.4375 # right, 7 / 16
                if (y + 1 < h) and (x + 1 < w):
                    image[y + 1, x + 1] += error * 0.0625 # right, down, 1 / 16
                if y + 1 < h:
                    image[y + 1, x] += error * 0.3125 # down, 5 / 16
                if (x - 1 >= 0) and (y + 1 < h): 
                    image[y + 1, x - 1] += error * 0.1875 # left, down, 3 / 16        
        return image
    
    @staticmethod
    def cutoff(image: np.ndarray, threshold=0.5, inplace=True):
        """
        Cutoff dithering algorithm
        --------------------
        Parameters:
        --------------------
        image: np.array of shape (height, width), dtype=float, 0.0-1.0
        threshold: float
            Threshold for the cutoff dithering algorithm
        --------------------
        Returns:
        --------------------
        image: np.array of shape (height, width), dtype=float, 0.0-1.0
        """
        if not inplace: image = image.copy()
        mask = image >= threshold
        image[mask] = 1
        image[~mask] = 0
        return image
    
    @staticmethod
    def random(image: np.ndarray, inplace=True):
        """
        Random dithering algorithm
        --------------------
        Parameters:
        --------------------
        image: np.array of shape (height, width), dtype=float, 0.0-1.0
        --------------------
        Returns:
        --------------------
        image: np.array of shape (height, width), dtype=float, 0.0-1.0
        """
        if not inplace: image = image.copy()
        mask = (image > np.random.random(image.shape))
        image[mask] = 1
        image[~mask] = 0
        return image

class Frame(object):
    def __init__(self, frame_type='binary', dmd_nrows=DMD_NROWS, dmd_ncols=DMD_NCOLS, flip=FLIP) -> None:
        """
        BasicFrame class is used to store the DMD image in a 2D array of values and perform coordinate conversion between real space and DMD space.
        --------------------
        Parameters:
        --------------------
        type: str
            Type of the frame, 'binary' for binary frame, 'gray' for grayscale frame, 'color' for RGB frame
        flip: bool
            True to flip the image vertically, False otherwise

        --------------------
        Attributes:
        --------------------
        real_frame: array-like
            The template image in real space, which is the image that will be converted to DMD space
        dmd_frame: array-like
            The DMD image in DMD space, which is the image that will be sent to the DMD
        flip: bool
            True to flip the image vertically (maybe necessary to get the right conversion), False otherwise
        dmd_nrows: int
            Number of rows in the DMD image
        dmd_ncols: int
            Number of columns in the DMD image
        real_nrows: int
            Number of rows in the real space image
        real_ncols: int
            Number of columns in the real space image
        DMD_NROWS: array-like of shape (N,)
            Row coordinates of the pixels in the real frame that are in the DMD frame
        DMD_NCOLS: array-like of shape (N,)
            Column coordinates of the pixels in the real frame that are in the DMD frame
        bg_rows: array-like of shape (M,)
            Row coordinates of the pixels in the real frame that are outside the DMD frame
        bg_cols: array-like of shape (M,)
            Column coordinates of the pixels in the real frame that are outside the DMD frame
        """
        self.flip = flip
        self.frame_type = frame_type
        self.dmd_nrows = dmd_nrows
        self.dmd_ncols = dmd_ncols
        self.real_nrows = math.ceil((self.dmd_nrows-1) / 2) + self.dmd_ncols
        self.real_ncols = self.dmd_ncols + (self.dmd_nrows-1) // 2

        # Generate a list of row and col coordinates in dmd space and find their corresponding in real space
        row, col = np.meshgrid(np.arange(self.dmd_nrows), np.arange(self.dmd_ncols), indexing='ij')
        self.DMD_NROWS, self.DMD_NCOLS = self.realSpace(row.flatten(), col.flatten())

        # Find the real space row and col coordinates of the pixels outside the DMD image
        mask = np.full((self.real_nrows, self.real_ncols), True, dtype=bool)
        mask[self.DMD_NROWS, self.DMD_NCOLS] = False
        real_row, real_col = np.meshgrid(np.arange(self.real_nrows), np.arange(self.real_ncols), indexing='ij')
        self.bg_rows, self.bg_cols = real_row[mask], real_col[mask]

        if self.bg_rows.shape[0] + self.DMD_NROWS.shape[0] != self.real_nrows * self.real_ncols:
            raise ValueError('Number of pixels in the DMD image does not match the number of pixels in the real space image')

        # Initialize the template image in real space and the DMD image in DMD space
        if self.frame_type == 'binary':
            self.real_frame = np.full((self.real_nrows, self.real_ncols), 0, dtype=bool)
            self.dmd_frame = np.full((self.dmd_nrows, self.dmd_ncols), 0, dtype=bool)
        elif self.frame_type == 'gray':
            self.real_frame = np.full((self.real_nrows, self.real_ncols), 0, dtype=np.float32)
            self.dmd_frame = np.full((self.dmd_nrows, self.dmd_ncols), 0, dtype=np.float32)
        elif self.frame_type == 'RGB':
            self.real_frame = np.full((self.real_nrows, self.real_ncols, 3), 0, dtype=np.uint8)
            self.dmd_frame = np.full((self.dmd_nrows, self.dmd_ncols, 3), 0, dtype=np.uint8)
        else:
            raise ValueError('Invalid frame type')
    
    def realSpace(self, row, col) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert the given DMD-space row and column to real-space row and column
        --------------------
        Parameters:
        --------------------
        row: int | array-like
            Row in DMD-space
        col: int | array-like
            Column in DMD-space
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
            real_row, real_col = (np.ceil((self.dmd_nrows - 1 - row)/2)).astype(int) + col, self.dmd_ncols - 1 + (self.dmd_nrows - 1 - row)//2 - col
        else:
            real_row, real_col = (np.ceil(row/2)).astype(int) + col, self.dmd_ncols - 1 + row//2 - col
        return real_row, real_col
    
    def parseColor(self, color, single=False):
        """
        Parse the color argument to a valid color (binary or 24-bit RGB)
        --------------------
        Parameters:
        --------------------
        color: int | float | array-like
            1 for white (on), 0 for black (off), float for grayscale, list or array-like of shape (3,) for RGB
            list or array-like of shape (N,) for binary/gray frame, list or array-like of shape (N, 3) for RGB frame
        
        --------------------
        Returns:
        --------------------
        color: parsed color
        """
        if self.frame_type == 'binary':
            if isinstance(color, int) and (color in [0, 1]):
                color = bool(color)
            elif isinstance(color, (list, tuple, np.ndarray)) and not single:
                color = np.array(color).astype(bool)
            else:
                raise ValueError('Invalid color for binary frame')

        elif self.frame_type == 'gray':
            if isinstance(color, (int, float)) and (color >= 0) and (color <= 1):
                color = np.clip(color, 0, 1).astype(np.float32)
            elif isinstance(color, (list, tuple, np.ndarray)) and not single:
                color = np.array(color).astype(np.float32)
                if ~(color.ndim == 1 and np.all(color >= 0) and np.all(color <= 1)):
                    raise ValueError('Invalid color for grayscale frame')
            else:
                raise ValueError('Invalid color for grayscale frame')

        elif self.frame_type == 'RGB':
            if isinstance(color, (float, int)):
                if (color >= 0 and color <= 1):
                    color = np.floor(255 * np.array([color, color, color])).astype(np.uint8)
                elif (color >= 0 and color <= 255):
                    color = np.floor([color, color, color]).astype(np.uint8)
                else:
                    raise ValueError('Invalid color for RGB frame')
            elif isinstance(color, (list, tuple, np.ndarray)):
                color = np.array(color).astype(np.uint8)
                if single and color.shape != (3,):
                    raise ValueError('Invalid color for RGB frame')
            else:
                raise ValueError('Invalid color for RGB frame')
            if ~(color.shape[-1] == 3 and np.all(color >= 0) and np.all(color <= 255)):
                    raise ValueError('Invalid color for RGB frame')
            
        return color

    def getFrameRGB(self):
        """
        Return the real-space and dmd-space frames as an RGB frame
        --------------------
        Returns:
        --------------------
        real_frame: array-like
            The real-space frame as an RGB frame
        """
        if self.frame_type == 'binary':
            real_frame = self.real_frame.astype(np.uint8) * 255
            real_frame = np.stack([real_frame] * 3, axis=-1)
            dmd_frame = self.dmd_frame.astype(np.uint8) * 255
            dmd_frame = np.stack([dmd_frame] * 3, axis=-1)
        elif self.frame_type == 'gray':
            real_frame = (self.real_frame * 255).astype(np.uint8)
            real_frame = np.stack([real_frame] * 3, axis=-1)
            dmd_frame = (self.dmd_frame * 255).astype(np.uint8)
            dmd_frame = np.stack([dmd_frame] * 3, axis=-1)
        elif self.frame_type == 'RGB':
            real_frame = self.real_frame
            dmd_frame = self.dmd_frame
        real_frame[self.bg_rows, self.bg_cols, :] = np.array([255, 0, 0])
        return real_frame, dmd_frame
    
    def fotmatTemplateImage(self, real_frame):
        """
        Return a PIL Image object of the real-space image with text labels on the corners
        --------------------
        Parameters:
        --------------------
        real_frame: array-like
            The real-space frame as an RGB frame
            
        --------------------
        Returns:
        --------------------
        template: PIL Image object
            The template image in real space
        """
        image = Image.fromarray(real_frame, mode='RGB')
        
        # Add labels on the corners
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 30)

        if self.flip:
            offset = ((150, -150), (0, 50), (150, 0))
        else:
            offset = ((0, -100), (150, -150), (-50, 50))

        # Coordinate of corners (col, row)
        corner00 = self.realSpace(0, 0)[1] + offset[0][1], self.realSpace(0, 0)[0] + offset[0][0]
        corner10 = self.realSpace(self.dmd_nrows-1, 0)[1] + offset[1][1], self.realSpace(self.dmd_nrows-1, 0)[0] + offset[1][0]
        corner11 = self.realSpace(self.dmd_nrows-1, self.dmd_ncols-1)[1] + offset[2][1], self.realSpace(self.dmd_nrows-1, self.dmd_ncols-1)[0] + offset[2][0]

        # Labels of corners (row, col)
        draw.text(corner00, '(0, 0)', font=font, fill=0)
        draw.text(corner10, f'({self.dmd_nrows-1}, 0)', font=font, fill=0)
        draw.text(corner11, f'({self.dmd_nrows-1}, {self.dmd_ncols-1})', font=font, fill=0)
        return image
    
    def displayPattern(self, 
                       real_space_title='Real-space Image', 
                       dmd_space_title='DMD-space Image'):
        """
        Display the real-space and DMD-space image
        --------------------
        Parameters:
        --------------------
        real_space_title: str
            Title of the real-space image, default is 'Real-space Image'
        dmd_space_title: str
            Title of the DMD-space image, default is 'DMD-space Image'
        """
        real_frame, dmd_frame = self.getFrameRGB()

        plt.subplots(figsize=(15, 30))
        plt.subplot(1, 2, 1)
        plt.imshow(real_frame)
        plt.box(True)
        plt.title(real_space_title)
        plt.subplot(1, 2, 2)
        plt.imshow(dmd_frame)
        plt.box(True)
        plt.title(dmd_space_title)
        plt.show()
    
    def displayRealSpaceImage(self):
        """
        Display the real-space image
        """
        real_frame, _ = self.getFrameRGB()
        image = Image.fromarray(real_frame, mode='RGB')
        image.show()        
    
    def updateDmdArray(self):
        """
        Update the DMD array from the real-space array
        """
        # Loop through every column and row for the DMD image and assign it 
        # the corresponding pixel value from the real space image
        if self.frame_type in ('binary', 'gray'):
            self.dmd_frame[:, :] = self.real_frame[self.DMD_NROWS, self.DMD_NCOLS].reshape(self.dmd_nrows, self.dmd_ncols)
        elif self.frame_type == 'RGB':
            self.dmd_frame[:, :, :] = self.real_frame[self.DMD_NROWS, self.DMD_NCOLS, :].reshape(self.dmd_nrows, self.dmd_ncols, 3)
    
    def saveFrameToFile(self, 
                        path: str, 
                        filename: str, 
                        save_template: bool=True, 
                        dmd_prefix='', 
                        template_prefix='',
                        separate_template_folder=True):
        """
        Save the DMD frame to a BMP file
        --------------------
        Parameters:
        --------------------
        path: str
            pathectory to save the BMP file
        filename: str
            Name of the BMP file to be saved
        save_template: bool
            True to save the real space template image, False otherwise
        """
        if os.path.exists(path) == False: os.makedirs(path)
        if separate_template_folder and os.path.exists(path + '/template/') == False: os.makedirs(path + '/template/')
        if separate_template_folder: temp_path = path + '/template/'
        if filename[-4:] != '.bmp': filename += '.bmp'

        dmd_filename = os.path.relpath(path + '/' + dmd_prefix + filename)
        template_filename = os.path.relpath(temp_path + '/' + template_prefix + filename)

        real_frame, dmd_frame = self.getFrameRGB()
        dmd_image = Image.fromarray(dmd_frame, mode='RGB')
        dmd_image.save(dmd_filename)
        print(f'DMD pattern saved as: .\{dmd_filename}')

        if save_template:
            template_image = self.fotmatTemplateImage(real_frame)
            template_image.save(template_filename, mode='RGB')
            print(f'Template image saved as: .\{template_filename}')
    
    def setRealArray(self, color=0):
        """
        Set the real-space array to a given color
        --------------------
        Parameters:
        --------------------
        color: int | array-like, color of the real-space array
            1 for white (on), 0 for black (off)
        """
        color = self.parseColor(color, single=True)
        self.real_frame[:, :] = color
        self.updateDmdArray()
    
    def drawPattern(self,
                    corr, 
                    color=1, 
                    reset=True, 
                    template_color=None, 
                    bg_color=0):
        """
        Draw a pattern on the real-space frame at the given coordinates
        --------------------
        Parameters:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the pattern
        color: int | array-like, color of the pattern
            1 for white (on), 0 for black (off)
        reset: bool
            True to reset the real space template to the default template, False otherwise
        template_color: int | array-like, color of the template
            1 for white (on), 0 for black (off)
        bg_color: int | array-like, color of the background
            1 for white (on), 0 for black (off)
        """
        # Reset the real space template
        color = self.parseColor(color)
        if reset: 
            if template_color is None:
                if isinstance(color, (int, bool)) and self.frame_type in ('binary', 'gray'):
                    template_color = 1 - color
                elif len(color) == 3 and self.frame_type == 'RGB':
                    template_color = 255 - color
                else:
                    template_color = 0
            self.setRealArray(color=template_color)
        
        # Draw a pattern on real-space array
        self.real_frame[corr[:, 0].astype(int), corr[:, 1].astype(int)] = color

        # Fill the background space with bg_color
        self.real_frame[self.bg_rows, self.bg_cols] = self.parseColor(bg_color)

        # Update the pixels in DMD space from the updated real-space array
        self.updateDmdArray()
    

class BinaryFrame(Frame):
    def __init__(self) -> None:
        """
        BinaryFrame class is used to store the binary image in a 2D array of boolean values and perform coordinate conversion between real space and DMD space.
        """
        super().__init__(frame_type='binary')

    def simulateImage(self, wavelength=532,):
        image = self.real_frame.sum(axis=2)
        image[self.bg_rows, self.bg_cols] = 0
    
    def convertToGrayFrame(self):
        """
        Convert the binary frame to a grayscale frame
        """
        gray_frame = GrayFrame()
        gray_frame.real_frame[:, :] = self.real_frame.astype(np.float32)
        gray_frame.updateDmdArray()
        return gray_frame
    
    def convertToColorFrame(self):
        """
        Convert the binary frame to an RGB frame
        """
        real_frame, dmd_frame = self.getFrameRGB()
        color_frame = ColorFrame()
        color_frame.real_frame[:, :, :] = real_frame
        color_frame.dmd_frame[:, :, :] = dmd_frame
        return color_frame


class GrayFrame(Frame):
    def __init__(self, ditcher='Floyd-Steinberg') -> None:
        """
        GrayFrame class is used to store the grayscale image in a 2D array of float values and perform coordinate conversion between real space and DMD space.
        """
        super().__init__(frame_type='gray')       
        self.binary_frame = BinaryFrame()
        if ditcher == 'Floyd-Steinberg':
            self.dither = Dither.floyd_steinberg
        elif ditcher == 'cutoff':
            self.dither = Dither.cutoff
        elif ditcher == 'random':
            self.dither = Dither.random
        else:
            raise ValueError('Invalid dithering algorithm')
    
    def drawPattern(self, corr, color=1, reset=True, template_color=None, bg_color=0):
        """
        Draw a pattern on the real-space frame at the given coordinates
        --------------------
        Parameters:
        --------------------
        corr: array-like of shape (N, 2) | (N, 3)
            Coordinates of the points in the pattern
        color: int | array-like, color of the pattern
            1 for white (on), 0 for black (off)
        reset: bool
            True to reset the real space template to the default template, False otherwise
        template_color: int | array-like, color of the template
            1 for white (on), 0 for black (off)
        bg_color: int | array-like, color of the background
            1 for white (on), 0 for black (off)
        """
        assert isinstance(corr, np.ndarray) and corr.ndim == 2, 'corr must be a 2D numpy array'
        if corr.shape[1] == 2:
            super().drawPattern(corr, color, reset, template_color, bg_color)
        elif corr.shape[1] == 3:
            super().drawPattern(corr[:, :2], corr[:, 2], reset, template_color, bg_color)
        else:
            raise ValueError('Invalid shape of corr, corr must be a 2D numpy array with shape (N, 2) or (N, 3)')
        self.ditherPattern()
    
    def ditherPattern(self):
        """
        Dither the real-space image to a binary image and update the DMD array
        """
        self.binary_frame.real_frame = self.dither(self.real_frame, inplace=False).astype(bool)
        self.binary_frame.updateDmdArray()

    def displayPattern(self):
        """
        Display the pattern with dithered binary image
        """
        plt.subplots(figsize=(30, 30))
        plt.subplot(2, 2, 1)
        plt.imshow(self.real_frame, cmap='gray')
        plt.box(True)
        plt.title('Real-space Image')
        plt.subplot(2, 2, 2)
        plt.imshow(self.binary_frame.real_frame, cmap='gray')
        plt.box(True)
        plt.title('Dithered Real-space Image')
        plt.subplot(2, 2, 3)
        plt.imshow(self.dmd_frame, cmap='gray')
        plt.box(True)
        plt.title('DMD-space Image')
        plt.subplot(2, 2, 4)
        plt.imshow(self.binary_frame.dmd_frame, cmap='gray')
        plt.box(True)
        plt.title('Dithered DMD-space Image')
        plt.show()

    def saveFrameToFile(self, path, filename, save_template=True, save_binary=True):
        """
        Save the DMD frame to a BMP file
        --------------------
        Parameters:
        --------------------
        path: str
            pathectory to save the BMP file
        filename: str
            Name of the BMP file to be saved
        save_template: bool
            True to save the real space template image, False otherwise
        save_binary: bool
            True to save the dithered binary image, False otherwise
        """
        super().saveFrameToFile(path, filename, save_template)
        if save_binary: self.binary_frame.saveFrameToFile(path + '/binary/', filename, save_template)

    def convertToBinaryFrame(self):
        """
        Convert the grayscale frame to a binary frame
        """
        self.ditherPattern()
        return self.binary_frame
    
    def convertToColorFrame(self):
        """
        Convert the grayscale frame to an RGB frame
        """
        real_frame, dmd_frame = self.getFrameRGB()
        color_frame = ColorFrame()
        color_frame.real_frame[:, :, :] = real_frame
        color_frame.dmd_frame[:, :, :] = dmd_frame
        return color_frame


class ColorFrame(Frame):
    def __init__(self) -> None:
        """
        ColorFrame class is used to store the RGB image in a 3D array of uint8 values and perform coordinate conversion between real space and DMD space.
        """
        super().__init__(frame_type='RGB')
    
    def loadFromFile(self, file_path):
        """
        Load a real-space image to the real-space array and convert it to DMD space
        --------------------
        Parameters:
        --------------------
        file_path: str
            Name of the image file with the path
        """
        image = Image.open(file_path).convert('RGB')
        self.real_frame[:, :, :] = np.asarray(image, dtype=np.uint8)
        self.updateDmdArray()

    def unpackFrames(self):
        """
        Unpack the real-space image to 24 binary frames
        """
        binary_frames = [BinaryFrame() for _ in range(24)]
        for i in range(3):
            real_frame = self.real_frame[:, :, i]
            dmd_frame = self.dmd_frame[:, :, i]
            for j in range(8):
                binary_frames[i*8+j].real_frame[:, :] = real_frame & (0b00000001 << j)
                binary_frames[i*8+j].dmd_frame[:, :] = dmd_frame & (0b00000001 << j)
        return binary_frames


class Painter(object):
    def __init__(self, nrows=REAL_NROWS, ncols=REAL_NCOLS) -> None:
        """
        Painter class is used to generate coordinates of patterns on a rectangular grid.
        The "draw" functions return the coordinates of the points in the pattern.
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
    
    def parseRange(self, x):
        """
        Parse the range argument to a valid range
        --------------------
        Parameters:
        --------------------
        x: int | array-like
            int for number of elements, array-like for list of elements

        --------------------
        Returns:
        --------------------
        x: array-like
            array-like of shape (N,)
        """
        if isinstance(x, int):
            assert x >= 0, 'x must be a non-negative integer'
            x = list(range(x))
        elif isinstance(x, range):
            x = list(x)
        return x

    def drawText(self, 
                    text='A', 
                    offset=(0, 0),
                    font_size=500, 
                    font='arial.ttf',
                    rotate=45):
        """
        Draw text on the rectangular grid
        --------------------
        Parameters:
        --------------------
        letter: str
            Letter to be drawn
        offset: tuple
            Offset of the center of the letter with respect to the center of the grid
        font_size: int
            Font size of the letter
        font: str
            Font of the letter
        rotate: float
            Angle to rotate the letter

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the letter
        """
        row, col = self.nrows // 2 + offset[0], self.ncols // 2 + offset[1]
        
        # Build a transparent image large enough to hold the text
        mask = Image.new('L', (self.nrows, self.ncols), 0)
        draw = ImageDraw.Draw(mask)
        font = ImageFont.truetype(font, font_size)
        draw.text((row, col), text=text, font=font, fill=255, anchor='mm')
        mask = mask.rotate(rotate)

        # Find the coordinates of the points in the letter
        return np.argwhere(np.array(mask) == 255)
    
    def drawCircle(self, 
                   row_offset=0, 
                   col_offset=0, 
                   radius=50):
        """
        Draw a filled circle on the rectangular grid
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
        nx = self.parseRange(nx)
        ny = self.parseRange(ny)
        corr = []
        for i, j in product(nx, ny):
            new_circle = self.drawCircle(row_offset=i*row_spacing+row_offset, 
                                    col_offset=j*col_spacing+col_offset, 
                                    radius=radius)
            if new_circle.shape[0] != 0: corr.append(new_circle)
        return np.concatenate(corr, axis=0)
    
    def drawHorizontalLine(self, 
                           row_offset=0, 
                           width=1,
                           center=False):
        """
        Draw a horizontal line on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            Row offset of the center of the line
        width: int
            Width of the line
        center: bool
            True to draw the line at the offseted center of the grid, False otherwise
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the line
        """
        # Find the center coordinates
        row = self.nrows // 2 + row_offset
        if center: 
            line_range = range(max(0, row - width // 2), min(self.nrows, row + width // 2 + 1))
        else:
            line_range = range(max(0, row), min(self.nrows, row + width))
        ans = np.array([(i, j) for j in range(self.ncols) for i in line_range])
        return ans
    
    def drawVerticalLine(self, 
                         col_offset=0, 
                         width=1, 
                         center=False):
        """
        Draw a vertical line on the rectangular grid
        --------------------
        Parameters:
        --------------------
        col_offset: int
            Column offset of the center of the line
        width: int
            Width of the line
        center: bool
            True to draw the line at the offseted center of the grid, False otherwise
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the line
        """
        # Find the center coordinates
        col = self.ncols // 2 + col_offset
        if center: 
            line_range = range(max(0, col - width // 2), min(self.ncols, col + width // 2 + 1))
        else:
            line_range = range(max(0, col), min(self.ncols, col + width))
        ans = np.array([(i, j) for i in range(self.nrows) for j in line_range])
        return ans
    
    def drawCross(self, 
                  row_offset=0, 
                  col_offset=0, 
                  half_width=1):
        """
        Draw a cross on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            Row offset of the center of the cross
        col_offset: int
            Column offset of the center of the cross
        half_width: int
            Half width of the lines in the cross

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the cross
        """
        return np.concatenate((self.drawHorizontalLine(row_offset=row_offset, width=2*half_width, center=True),
                               self.drawVerticalLine(col_offset=col_offset, width=2*half_width, center=True)), axis=0)
    
    def drawHorizontalLines(self, 
                            row_spacing=50, 
                            row_offset=0, 
                            half_width=1, 
                            ny=5):
        """
        Draw an array of horizontal lines on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_spacing: int
            Spacing between rows of lines
        row_offset: int
            Row offset of the center of the first line
        half_width: int
            Half width of the lines
        ny: int | array-like
            Number of lines, or a list of row indices to draw lines on
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of lines
        """
        ny = self.parseRange(ny)
        corr = []
        for i in ny:
            new_line = self.drawHorizontalLine(row_offset=i*row_spacing+row_offset, 
                                        width=2*half_width,
                                        center=True)
            if new_line.shape[0] != 0: corr.append(new_line)
        return np.concatenate(corr, axis=0)
    
    def drawVerticalLines(self, 
                          col_spacing=50, 
                          col_offset=0, 
                          half_width=1, 
                          nx=5):
        """
        Draw an array of vertical lines on the rectangular grid
        --------------------
        Parameters:
        --------------------
        col_spacing: int
            Spacing between columns of lines
        col_offset: int
            Column offset of the center of the first line
        half_width: int
            Half width of the lines
        nx: int | array-like
            Number of lines, or a list of column indices to draw lines on
                  
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of lines
        """
        nx = self.parseRange(nx)
        corr = []
        for j in nx:
            new_line = self.drawVerticalLine(col_offset=j*col_spacing+col_offset, 
                                        width=2*half_width,
                                        center=True)
            if new_line.shape[0] != 0: corr.append(new_line)
        return np.concatenate(corr, axis=0)
    
    def drawAngledLine(self, 
                       angle=45, 
                       row_offset=0, 
                       col_offset=0, 
                       width=10,
                       center=False):
        """
        Draw an angled line on the rectangular grid
        --------------------
        Parameters:
        --------------------
        angle: int
            Angle of the line in degrees
        row_offset: int
            Row offset of the center of the line
        col_offset: int
            Column offset of the center of the line
        width: int
            Width of the line

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the line
        """
        angle = angle % 180
        if angle == 0:
            return self.drawHorizontalLine(row_offset=row_offset, width=width, center=center)
        elif angle == 90:
            return self.drawVerticalLine(col_offset=col_offset, width=width, center=center)
        
        # Find the center coordinates
        center_row, center_col = self.nrows // 2 + row_offset, self.ncols // 2 + col_offset
        
        # Draw a line with the given angle
        angle = np.deg2rad(angle)
        rows, cols = np.meshgrid(np.arange(self.nrows), np.arange(self.ncols), indexing='ij')
        dist = (cols - center_col) * np.sin(angle) - (rows - center_row) * np.cos(angle)
        if center:
            mask = (np.abs(dist) <= width // 2).astype(bool).flatten()
        else:
            mask = ((dist >= 0) & (dist <= width)).astype(bool).flatten()
        
        return np.stack((rows.flatten()[mask], cols.flatten()[mask])).transpose()
    
    def drawCrosses(self, 
                    row_spacing=50, 
                    col_spacing=50, 
                    row_offset=0, 
                    col_offset=0, 
                    half_width=1, 
                    nx=5, 
                    ny=5):
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
        half_width: int
            Half width of the lines in the crosses
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
        nx = self.parseRange(nx)
        ny = self.parseRange(ny)
        corr = [self.drawHorizontalLines(row_spacing=row_spacing, row_offset=row_offset, half_width=half_width, ny=ny),
                self.drawVerticalLines(col_spacing=col_spacing, col_offset=col_offset, half_width=half_width, nx=nx)]
        return np.concatenate(corr, axis=0)
    
    def drawAngledCross(self,
                        angle=45,
                        row_offset=0,
                        col_offset=0,
                        half_width=10):
        """
        Draw an angled cross on the rectangular grid
        --------------------
        Parameters:
        --------------------
        angle: float
            Angle of the cross in degrees
        row_offset: int
            Row offset of the center of the cross
        col_offset: int
            Column offset of the center of the cross
        half_width: int
            Half width of the lines in the cross

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the cross
        """
        return np.concatenate((self.drawAngledLine(angle=angle, row_offset=row_offset, col_offset=col_offset, width=2*half_width, center=True),
                               self.drawAngledLine(angle=angle+90, row_offset=row_offset, col_offset=col_offset, width=2*half_width, center=True)), axis=0)

    def drawStar(self, 
                 row_offset=0, 
                 col_offset=0, 
                 num=10):
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
        center_row, center_col = self.nrows // 2 + row_offset, self.ncols // 2 + col_offset

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
        mask = (((rows.flatten() // size) % 2 + (cols.flatten() // size) % 2) % 2).astype(bool)
        return np.stack((rows.flatten()[mask], cols.flatten()[mask])).transpose()
    
    def drawSquare(self, 
                   radius=3, 
                   row_offset=0, 
                   col_offset=0):
        """
        Draw a square on the rectangular grid
        --------------------
        Parameters:
        --------------------
        radius: int
            Radius of the square
        row_offset: int
            Row offset of the center of the square
        col_offset: int
            Column offset of the center of the square

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the square
        """
        center_row, center_col = self.nrows // 2 + row_offset, self.ncols // 2 + col_offset
        ans = [(i, j) for i in range(max(0, center_row - radius), min(center_row + radius + 1, self.nrows))\
                for j in range(max(0, center_col - radius), min(center_col + radius + 1, self.ncols))]
        return np.array(ans).astype(int)
    
    def drawArrayOfSquares(self, 
                           row_spacing=50, 
                           col_spacing=50, 
                           row_offset=0, 
                           col_offset=0, 
                           nx=5, 
                           ny=5, 
                           radius=3):
        """
        Draw an array of squares on the rectangular grid

        --------------------
        Parameters:
        --------------------
        row_spacing: int
            Spacing between rows of squares
        col_spacing: int
            Spacing between columns of squares
        row_offset: int
            Row offset of the center of the first square
        col_offset: int
            Column offset of the center of the first square
        nx: int
            Number of squares in each row
        ny: int
            Number of squares in each column
        radius: int
            Radius of the squares

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of squares
        """
        nx = self.parseRange(nx)
        ny = self.parseRange(ny)
        
        corr = []
        for i, j in product(nx, ny):
            new_square = self.drawSquare(row_offset=i*row_spacing+row_offset, 
                                    col_offset=j*col_spacing+col_offset, 
                                    radius=radius)
            if new_square.shape[0] != 0: corr.append(new_square)
        return np.concatenate(corr, axis=0)
    
    def drawHorizontalStrips(self, 
                             width=5, 
                             row_offset=0):
        """
        Draw an array of horizontal strips on the rectangular grid
        --------------------
        Parameters:
        --------------------
        width: int
            Width of the strips
        row_offset: int
            Row offset of the top of the first strip

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of strips
        """
        corr = [self.drawHorizontalLine(row_offset=i-self.nrows//2+row_offset, width=width) for i in range(0, self.nrows - width, 2*width)]
        return np.concatenate(corr, axis=0)
    
    def drawVerticalStrips(self, 
                           width=5, 
                           col_offset=0):
        """
        Draw an array of vertical strips on the rectangular grid
        --------------------
        Parameters:
        --------------------
        width: int
            Width of the strips
        col_offset: int
            Column offset of the left of the first strip

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of strips
        """
        corr = [self.drawVerticalLine(col_offset=j-self.ncols//2+col_offset, width=width) for j in range(0, self.ncols - width, 2*width)]
        return np.concatenate(corr, axis=0)

    def drawAngledStrips(self,
                        angle=45,
                        width=5,
                        row_offset=0,
                        col_offset=0):
        """
        Draw an array of angled strips on the rectangular grid
        --------------------
        Parameters:
        --------------------
        angle: float
            Angle of the strips in degrees
        width: int
            Width of the strips
        row_offset: int
            Row offset of the top of the first strip
        col_offset: int
            Column offset of the left of the first strip

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of strips
        """
        angle = angle % 180
        if angle == 0:
            return self.drawHorizontalStrips(width=width, row_offset=row_offset)
        elif angle == 90:
            return self.drawVerticalStrips(width=width, col_offset=col_offset)
        
        # Find the center coordinates
        center_row, center_col = self.nrows // 2 + row_offset, self.ncols // 2 + col_offset

        # Draw a line with the given angle
        angle = np.deg2rad(angle)
        rows, cols = np.meshgrid(np.arange(self.nrows), np.arange(self.ncols), indexing='ij')
        dist = (cols - center_col) * np.sin(angle) - (rows - center_row) * np.cos(angle)
        mask = (dist % (2*width) <= width).astype(bool).flatten()

        return np.stack((rows.flatten()[mask], cols.flatten()[mask])).transpose()

    
    def drawHorizontalHalfPlane(self,
                                row_offset=0):
        """
        Draw a horizontal half plane on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            row offset of the half plane
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the half plane
        """
        center_row = self.nrows // 2 + row_offset
        assert center_row >= 0 and center_row < self.nrows, 'Row offset out of range'
        ans = np.array([(i, j) for i in range(center_row, self.nrows) for j in range(self.ncols)])
        return ans

    def drawVerticalHalfPlane(self,
                              col_offset=0):
        """
        Draw a vertical half plane on the rectangular grid
        --------------------
        Parameters:
        --------------------
        col_offset: int
            column offset of the half plane

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the half plane
        """
        center_col = self.ncols // 2 + col_offset
        assert center_col >= 0 and center_col < self.ncols, 'Column offset out of range'
        ans = np.array([(i, j) for i in range(self.nrows) for j in range(center_col, self.ncols)])
        return ans
    
    def drawAnchorCircles(self,
                          anchor=((0, 0), (200, 0), (0, 250)),
                          radius=10):
        """
        Draw anchor circles on the rectangular grid
        --------------------
        Parameters:
        --------------------
        anchor: array-like of shape (N, 2)
            coordinates of the anchor circles
        radius: int
            radius of the anchor circles

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the anchor circles
        """
        corr = []
        for x, y in anchor:
            corr.append(self.drawCircle(row_offset=x, 
                                        col_offset=y, 
                                        radius=radius))
        return np.concatenate(corr, axis=0)

    def drawAnchorCirclesWithBackgroundCircles(self, 
                                                bg_spacing=50,
                                                bg_radius=2,
                                                anchor=((0, 0), (100, 0), (0, 150)),
                                                anchor_radius=5):
        """
        Draw anchor circles with background circles on the rectangular grid
        --------------------
        Parameters:
        --------------------
        bg_spacing: int
            Spacing between rows and columns of background circles
        bg_radius: int
            Radius of the background circles
        anchor: array-like of shape (N, 2)
            coordinates of the anchor circles
        anchor_radius: int
            Radius of the anchor circles

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the anchor circles with background circles
        """
        corr = [self.drawArrayOfCircles(row_spacing=bg_spacing, 
                                        col_spacing=bg_spacing, 
                                        row_offset=0, 
                                        col_offset=0, 
                                        nx=range(-20, 20), 
                                        ny=range(-20, 20), 
                                        radius=bg_radius),
                self.drawAnchorCircles(anchor=anchor, radius=anchor_radius)]
        return np.concatenate(corr, axis=0)
    
    def drawAnchorCirclesWithBackgroundGrid(self, 
                                            bg_spacing=50,
                                            bg_halfwidth=5,
                                            anchor=((0, 0), (200, 0), (0, 250)),
                                            anchor_radius=5):
        """
        Draw anchor circles with background grid on the rectangular grid
        --------------------
        Parameters:
        --------------------
        bg_spacing: int
            Spacing between rows and columns of the background grid
        bg_halfwidth: int
            Half width of the lines in the background grid
        anchor: array-like of shape (N, 2)
            coordinates of the anchor circles
        anchor_radius: int
            Radius of the anchor circles

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the anchor circles with background grid
        """
        corr = [self.drawHorizontalLines(row_spacing=bg_spacing,
                                        row_offset=0,
                                        half_width=bg_halfwidth,
                                        ny=range(-20, 20)),
                self.drawVerticalLines(col_spacing=bg_spacing,
                                        col_offset=0,
                                        half_width=bg_halfwidth,
                                        nx=range(-20, 20)),
                self.drawAnchorCircles(anchor=anchor, radius=anchor_radius)]
        return np.concatenate(corr, axis=0)
    

class GrayscalePainter(Painter):
    def __init__(self, nrows=REAL_NROWS, ncols=REAL_NCOLS) -> None:
        super().__init__(nrows, ncols)
        self.rows, self.cols = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing='ij')
        self.rows, self.cols = self.rows.flatten().astype(int), self.cols.flatten().astype(int)

    def checkVector(self, vector):
        """
        Check if the given vector is a valid lattice vector
        --------------------
        Parameters:
        --------------------
        vector: list | array-like
            Lattice vector to be checked

        --------------------
        Raises:
        --------------------
        ValueError: if the given vector is not a valid lattice vector
        """
        if isinstance(vector, list):
            assert len(vector) == 2, 'Lattice vector 1 must be a list of length 2'
        elif isinstance(vector, np.ndarray):
            assert vector.shape == (2,), 'Lattice vector 1 must be an array of shape (2,)'
        else:
            raise ValueError('Lattice vector 1 must be a list or numpy array of shape (2,)')

    def draw1dLattice(self,
                      lat_vec=[0.01, 0.01],
                      x_offset=0,
                      y_offset=0,
                      ):
        """
        Draw a 1D lattice on the rectangular grid. The intensity is given by cos(2*\pi*k*x) where k is the lattice vector.
        --------------------
        Parameters:
        --------------------
        lat_vec: array-like of shape (2,)
            Lattice vector of the lattice
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 3)
            Coordinates of the points in the lattice, the third column as the intensity
        """
        self.checkVector(lat_vec)
        center_row, center_col = self.nrows // 2 + x_offset, self.ncols // 2 + y_offset
        pattern = np.cos(2 * np.pi * (lat_vec[0]*(self.rows - center_row) + lat_vec[1]*(self.cols - center_col)))
        Dither.normalizePattern(pattern)
        return np.stack((self.rows, self.cols, pattern.flatten())).transpose()
    
    def draw2dLattice(self,
                      lat_vec1 = [0.01, 0.],
                      lat_vec2 = [0., 0.01],
                      x_offset=0, 
                      y_offset=0,
                      interference=False,
                      ):
        """
        Draw a 2D lattice on the rectangular grid. The intensity is given by cos(k1*x) + cos(k2*x) + {cos((k1-k2)*x)} where k1, k2 is the lattice vector.
        --------------------
        Parameters:
        --------------------
        lat_vec1: array-like of shape (2,)
            Lattice vector of the first lattice beam
        lat_vec2: array-like of shape (2,)
            Lattice vector of the second lattice beam
        x_offset: int
            Row offset of the center of the lattice
        y_offset: int
            Column offset of the center of the lattice
        interference: bool
            True to add interference term to the lattice

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 3)
            Coordinates of the points in the lattice, the third column as the intensity
        """
        self.checkVector(lat_vec1)
        self.checkVector(lat_vec2)
        center_row, center_col = self.nrows // 2 + x_offset, self.ncols // 2 + y_offset

        if not interference:
            pattern = np.cos(2 * np.pi * (lat_vec1[0]*(self.rows - center_row) + lat_vec1[1]*(self.cols - center_col))) + \
                        np.cos(2 * np.pi * (lat_vec2[0]*(self.rows - center_row) + lat_vec2[1]*(self.cols - center_col)))
        else:
            pattern = np.cos(2 * np.pi * (lat_vec1[0]*(self.rows - center_row) + lat_vec1[1]*(self.cols - center_col))) + \
                        np.cos(2 * np.pi * (lat_vec2[0]*(self.rows - center_row) + lat_vec2[1]*(self.cols - center_col))) + \
                        np.cos(2 * np.pi * ((lat_vec1[0] - lat_vec2[0])*(self.rows - center_row) + (lat_vec1[1] - lat_vec2[1])*(self.cols - center_col)))
        Dither.normalizePattern(pattern)
        return np.stack((self.rows, self.cols, pattern.flatten())).transpose()


class BinarySequence(object):
    def __init__(self, nframes: int=24) -> None:
        """
        BinarySequence class is used to store a sequence of binary frames and perform operations on the sequence.
        --------------------
        Parameters:
        --------------------
        nframes: int
            Number of binary frames in the sequence
        """
        assert isinstance(nframes, int) and nframes > 0, 'Number of frames must be a positive integer'
        self.frames = [BinaryFrame() for _ in range(nframes)]
        self.RGB_frames = None
        self.nframes = nframes
        self.dmd_nrows, self.dmd_ncols = self.frames[0].dmd_frame.shape
        self.real_nrows, self.real_ncols = self.frames[0].real_frame.shape
    
    def packFrames(self):
        """
        Pack the binary frames to a list of RGB color frame
        """
        self.RGB_frames = [ColorFrame() for _ in range(math.ceil(self.nframes / 24))]
        for i in range(self.nframes):
            i_RGB = i // 24
            i_color = (i % 24) // 8
            i_bit = (i % 24) % 8
            self.RGB_frames[i_RGB].real_frame[:, :, i_color] = self.RGB_frames[i_RGB].real_frame[:, :, i_color] + self.frames[i].real_frame[:, :] * (0b00000001 << i_bit)
            self.RGB_frames[i_RGB].dmd_frame[:, :, i_color] = self.RGB_frames[i_RGB].dmd_frame[:, :, i_color] + self.frames[i].dmd_frame[:, :] * (0b00000001 << i_bit)

    def drawPatternOnFrame(self, i, corr, color=1, reset=True, template_color=None):
        """
        Draw a pattern on the binary frame
        --------------------
        Parameters:
        --------------------
        i: int
            Index of the binary frame to draw the pattern
        corr: array-like of shape (N, 2)
            Coordinates of the points in the pattern
        color: int
            Color of the pattern, black or white (0 or 1)
        reset: bool
            True to reset the frame before drawing the pattern
        template_color: int
            Color of the template, black or white (0 or 1)
        """
        assert isinstance(i, int) and i >= 0 and i < self.nframes, 'Index out of range'
        assert isinstance(corr, np.ndarray) and corr.shape[1] == 2, 'Invalid coordinates'
        self.frames[i].drawPattern(corr, color, reset, template_color)

    def saveRGBFrames(self, path: str, filename: str):
        """
        Save the packed RGB frames to a folder
        --------------------
        Parameters:
        --------------------
        path: str
            Path to the folder to save the RGB frames
        filename: str
            Name of the RGB frames
        """
        for i, frame in enumerate(self.RGB_frames):
            frame.saveFrameToFile(path, f'RGB_{i+1}_' + filename)

    def saveBinaryFrames(self, path: str, filename: str):
        """
        Save the binary frames to a folder
        --------------------
        Parameters:
        --------------------
        path: str
            Path to the folder to save the binary frames
        filename: str
            Name of the binary frames
        """
        for i, frame in enumerate(self.frames):
            frame.saveFrameToFile(path, f'Binary_{i+1}_'+ filename)

    def displayRGBFrames(self, start=0, end=None):
        """
        Display the packed RGB frames
        """
        if self.RGB_frames is None:
            self.packFrames()
        if end is None:
            end = len(self.RGB_frames)
        for i in range(start, end):
            self.RGB_frames[i].displayPattern(real_space_title=f'RGB_{i+1} Real Space', dmd_space_title=f'RGB_{i+1} DMD Space')
    
    def displayBinaryFrames(self, start=0, end=None):
        """
        Display the binary frames
        """
        if end is None:
            end = self.nframes
        for i in range(start, end):
            self.frames[i].displayPattern(real_space_title=f'Binary_{i+1} Real Space', dmd_space_title=f'Binary_{i+1} DMD Space')