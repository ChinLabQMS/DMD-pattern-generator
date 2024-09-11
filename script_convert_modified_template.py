import os, logging
from tkinter.filedialog import askopenfilename, asksaveasfilename
from core import ColorFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

file_path = askopenfilename(title='Select modified BMP Image template', filetypes=[('BMP Files', '*.bmp')],
                            initialdir=os.getcwd())
directory, filename = os.path.split(file_path)

# Convert the loaded modified template to a DMD Image
dmd_image = ColorFrame()
dmd_image.loadFromFile(file_path)

# Show the converted DMD pattern and save it to your directory
save_path = asksaveasfilename(title='Select directory to save the new BMP Image',
                              filetypes=[('BMP Files', '*.bmp')],
                              initialdir=directory, initialfile=filename)
directory, filename = os.path.split(save_path)
dmd_image.saveFrameToFile(directory, filename, save_template=False)
dmd_image.displayDmdSpaceImage()
