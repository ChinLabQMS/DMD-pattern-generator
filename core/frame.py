from PIL import Image, ImageDraw, ImageFont
import math, os, logging
import numpy as np
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


logger = logging.getLogger(__name__)

class Dither(object):
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
            if not(color.shape[-1] == 3 and np.all(color >= 0) and np.all(color <= 255)):
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
    
    def formatTemplateImage(self, 
                            real_frame=None, 
                            note=None, 
                            corner_label: bool=True):
        """
        Return a PIL Image object of the real-space image with text labels on the corners
        --------------------
        Parameters:
        --------------------
        real_frame: array-like
            The real-space frame as an RGB frame
        note: str
            Note to be added to the image
        corner_label: bool
            True to add labels on the corners, False otherwise
            
        --------------------
        Returns:
        --------------------
        template: PIL Image object
            The template image in real space
        """
        if real_frame is None: real_frame, _ = self.getFrameRGB()

        image = Image.fromarray(real_frame, mode='RGB')        
        draw = ImageDraw.Draw(image)

        # Add labels on the corners
        if corner_label:                 
            offset = ((150, -150), (0, 50), (150, 0)) if self.flip else ((0, -100), (150, -150), (-50, 50))

            # Coordinate of corners (col, row)
            corner00 = self.realSpace(0, 0)[1] + offset[0][1], self.realSpace(0, 0)[0] + offset[0][0]
            corner10 = self.realSpace(self.dmd_nrows-1, 0)[1] + offset[1][1], self.realSpace(self.dmd_nrows-1, 0)[0] + offset[1][0]
            corner11 = self.realSpace(self.dmd_nrows-1, self.dmd_ncols-1)[1] + offset[2][1], self.realSpace(self.dmd_nrows-1, self.dmd_ncols-1)[0] + offset[2][0]

            # Labels of corners (row, col)
            font = ImageFont.truetype("arial.ttf", 30)
            draw.text(corner00, '(0, 0)', font=font, fill=0)
            draw.text(corner10, f'({self.dmd_nrows-1}, 0)', font=font, fill=0)
            draw.text(corner11, f'({self.dmd_nrows-1}, {self.dmd_ncols-1})', font=font, fill=0)

        # Add note to the image
        if note is not None:
            draw.text((100, 100), note, font=ImageFont.truetype("arial.ttf", 80), fill=0)
        
        return image
    
    def displayPattern(self, 
                       real_space_title='Real-space image', 
                       dmd_space_title='DMD-space image'):
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

        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(real_frame)
        plt.title(real_space_title)
        plt.subplot(1, 2, 2)
        plt.imshow(dmd_frame)
        plt.title(dmd_space_title)
        plt.show()
    
    def displayRealSpaceImage(self):
        """
        Display the real-space image
        """
        real_frame, _ = self.getFrameRGB()
        image = Image.fromarray(real_frame, mode='RGB')
        image.show()

    def displayDmdSpaceImage(self):
        """
        Display the DMD-space image
        """
        _, dmd_frame = self.getFrameRGB()
        image = Image.fromarray(dmd_frame, mode='RGB')
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
                        template_note=None,
                        template_corner_label=True,
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
        if os.path.exists(path) == False: 
            os.makedirs(path)
        if separate_template_folder and save_template and os.path.exists(path + '/template/') == False: 
            os.makedirs(path + '/template/')
        temp_path = path if not separate_template_folder else path + '/template/'
        if not(filename.endswith('.bmp')): filename += '.bmp'

        dmd_filename = os.path.relpath(path + '/' + dmd_prefix + filename)
        template_filename = os.path.relpath(temp_path + '/' + template_prefix + filename)

        real_frame, dmd_frame = self.getFrameRGB()
        dmd_image = Image.fromarray(dmd_frame, mode='RGB')
        dmd_image.save(dmd_filename)
        logger.info(f'DMD pattern saved as: .\{dmd_filename}')

        if save_template:
            template_image = self.formatTemplateImage(real_frame, note=template_note, corner_label=template_corner_label)
            template_image.save(template_filename, mode='RGB')
            logger.info(f'Template image saved as: .\{template_filename}')
    
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
                    corr=None, 
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
        if corr is None: return
        
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
    def __init__(self, dither='Floyd-Steinberg') -> None:
        """
        GrayFrame class is used to store the grayscale image in a 2D array of float values and perform coordinate conversion between real space and DMD space.
        """
        super().__init__(frame_type='gray')       
        self.binary_frame = BinaryFrame()
        if dither == 'Floyd-Steinberg':
            self.dither = Dither.floyd_steinberg
        elif dither == 'cutoff':
            self.dither = Dither.cutoff
        elif dither == 'random':
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
        assert image.size == (self.real_ncols, self.real_nrows), \
            f'Image size does not match the frame size, please resize the image to {self.real_nrows}x{self.real_ncols} pixels'
        self.real_frame[:, :, :] = np.asarray(image, dtype=np.uint8)
        self.updateDmdArray()

    def unpackFramesToSequence(self, format='GRB'):
        """
        Unpack the real-space image to 24 binary frames and use a BinarySequence object to store the frames
        --------------------
        Parameters:
        --------------------
        format: str
            Format of the packing, 'RGB' for 24-bit R-G-B, 'GRB' for 24-bit G-R-B packing order
        """
        sequence = BinarySequence(nframes=24, packing_format=format)
        binary_frames = sequence.frames
        color_order = [1, 0, 2] if format == 'GRB' else [0, 1, 2]
        for i, color in enumerate(color_order):
            real_frame = self.real_frame[:, :, color]
            dmd_frame = self.dmd_frame[:, :, color]
            for j in range(8):
                binary_frames[i*8+j].real_frame[:, :] = real_frame & (0b00000001 << j)
                binary_frames[i*8+j].dmd_frame[:, :] = dmd_frame & (0b00000001 << j)
        return sequence

class BinarySequence(object):
    def __init__(self, nframes: int=24, packing_format='GRB') -> None:
        """
        BinarySequence class is used to store a sequence of binary frames and perform operations on the sequence.
        --------------------
        Parameters:
        --------------------
        nframes: int
            Number of binary frames in the sequence
        packing_format: str
            Format of the packing, 'RGB' for 24-bit R-G-B, 'GRB' for 24-bit G-R-B packing order
        """
        assert isinstance(nframes, int) and nframes > 0, 'Number of frames must be a positive integer'
        self.frames = [BinaryFrame() for _ in range(nframes)]
        self.color_frames = None
        self.nframes = nframes
        self.packing_format = packing_format
        self.dmd_nrows, self.dmd_ncols = self.frames[0].dmd_frame.shape
        self.real_nrows, self.real_ncols = self.frames[0].real_frame.shape
    
    def packFramesToColor(self, format='GRB', last_black:bool=False):
        """
        Pack the binary frames to a list of color frame
        --------------------
        Parameters:
        --------------------
        format: str
            Format of the packing, 'RGB' for 24-bit R-G-B, 'GRB' for 24-bit G-R-B packing order
        last_black: bool
            True to set the last frame of each 24-bit frame to black, False otherwise
        """
        self.packing_format = format
        self.color_frames = [ColorFrame() for _ in range(math.ceil(self.nframes / 24))]
        color_order = [1, 0, 2] if format == 'GRB' else [0, 1, 2]
        for i in range(self.nframes):
            i_frame = i // 24
            i_color = color_order[(i % 24) // 8]
            i_bit = (i % 24) % 8
            if last_black and (i % 24 == 23) : continue
            self.color_frames[i_frame].real_frame[:, :, i_color] = self.color_frames[i_frame].real_frame[:, :, i_color] + self.frames[i].real_frame[:, :] * (0b00000001 << i_bit)
            self.color_frames[i_frame].dmd_frame[:, :, i_color] = self.color_frames[i_frame].dmd_frame[:, :, i_color] + self.frames[i].dmd_frame[:, :] * (0b00000001 << i_bit)

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

    def saveRGBFrames(self, 
                      path:str, 
                      filename:str, 
                      index:int=None, 
                      index_str:str=None,
                      save_template:bool=True,
                      save_last_black:bool=True):
        """
        Save the packed RGB frames to a folder
        --------------------
        Parameters:
        --------------------
        path: str
            Path to the folder to save the RGB frames
        filename: str
            Name of the RGB frames
        index: int or None
            Index of the RGB frames to save, None to save all frames
        index_str: str or None
            Index string to be added to the filename
        save_template: bool
            True to save the real-space template image, False to only save DMD-space image
        save_last_black: bool
            True to save a copy that has the last frame of each 24-bit frame to black, False otherwise
        """
        last_black = [False, True] if save_last_black else [False]
        for b in last_black:
            self.packFramesToColor(last_black=b)
            filename = os.path.splitext(filename)[0] + ('_last_black' if b else '') + '.bmp'
            if index is not None:
                if index_str is None: index_str = str(index + 1)
                self.color_frames[index].saveFrameToFile(path, f'{self.packing_format}_{index_str}_' + filename, 
                                                        save_template=save_template)
            else:
                for i, frame in enumerate(self.color_frames):
                    frame.saveFrameToFile(path, f'{self.packing_format}_{i+1}_' + filename,
                                        save_template=save_template)

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
            frame.saveFrameToFile(path, f'Binary_{i+1}_'+ filename, template_corner_label=False, template_note=f'Frame: {i+1} / {self.nframes}')

    def saveSequenceToGIF(self, path: str, filename: str, duration=200):
        """
        Save the sequence of binary frames to a GIF file
        --------------------
        Parameters:
        --------------------
        path: str
            Path to the folder to save the GIF file
        filename: str
            Name of the GIF file
        duration: int
            Duration of each frame in milliseconds
        """
        if filename[-4:] != '.gif': filename += '.gif'
        if os.path.exists(path) == False: os.makedirs(path)
        filename = os.path.relpath(path + '/' + filename)

        images = []
        for i, frame in enumerate(self.frames):
            image = frame.formatTemplateImage(corner_label=False, note=f'Frame: {i+1} / {self.nframes}')
            images.append(image)
        images[0].save(filename, save_all=True, append_images=images[1:], duration=duration)
        logger.info(f'GIF file saved as: .\{filename}')

    def displayRGBFrames(self):
        """
        Display the packed RGB frames
        """
        if self.color_frames is None: self.packFramesToColor()
        for i in range(len(self.color_frames)):
            self.color_frames[i].displayPattern(real_space_title=f'RGB_{i+1} Real Space', dmd_space_title=f'RGB_{i+1} DMD Space')
    
    def displayBinaryFrames(self):
        """
        Display the binary frames
        """
        for i in range(self.nframes):
            if i % 24 == 0: plt.figure(figsize=(12, 8))
            image, _ = self.frames[i].getFrameRGB()
            plt.subplot(4, 6, i % 24 + 1)
            plt.imshow(image)
            plt.title(f'Frame {i+1}')
            plt.axis('off')
            if i % 24 == 23: plt.show()
