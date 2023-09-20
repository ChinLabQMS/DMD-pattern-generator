# DMD-pattern-generator
Generate patterns for calibrate DMD projected images.

## Usage
First, install dependencies: `pip3 install -r requirements.txt`

There are two operating modes:
* Generating patterns from modified template: open a terminal window and run `python3 main.py`, use the pop-up window to select modified template file, and the generated patterns will be saved in the same directory as the template file
* Generating patterns programmatically: there are a few pre-defined patterns written in [utils.py](utils.py), and can be used to generate regular patterns. Demo code is in this Jupyter notebook [DMD_pattern_generator.ipynb](DMD_pattern_generation.ipynb).