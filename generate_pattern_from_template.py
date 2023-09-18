import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from PIL import Image
from utils import DMDImage

if __name__ == '__main__':

    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title='Select modified BMP Image template', filetypes=[('BMP Files', '*.bmp')])
    directory, filename = os.path.split(file_path)
    filename_new = 'DMD_pattern_' + os.path.splitext(filename)[0] + '.bmp'

    # Convert the loaded modified template to a DMD Image
    dmd_image = DMDImage()
    dmd_image.convertImageToDMDArray(Image.open(file_path).convert('RGB'))

    # Show the converted DMD pattern and save it to your directory
    dmd_image.saveDMDArray(os.path.join(directory, filename_new))