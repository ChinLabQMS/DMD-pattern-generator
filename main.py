import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from utils import ColorFrame

if __name__ == '__main__':

    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title='Select modified BMP Image template', filetypes=[('BMP Files', '*.bmp')],
                                initialdir=os.getcwd())
    directory, filename = os.path.split(file_path)
    filename_new = os.path.splitext(filename)[0] + '_new.bmp'

    # Convert the loaded modified template to a DMD Image
    dmd_image = ColorFrame()
    dmd_image.loadFromFile(file_path)

    # Show the converted DMD pattern and save it to your directory
    dmd_image.saveFrameToFile(directory, filename_new, save_template=False)
    dmd_image.displayDmdSpaceImage()