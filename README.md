# DMD-pattern-generator
Generate calibration patterns for projector-based structured light system.
The code is developed with python3.10 and tested on windows 10 and 11.

## Dependencies
First, install dependencies, open a terminal under the project directory and run: 
```
pip3 install -r requirements.txt
```
It is recommended to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) or [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for running this code.

## Usage
There are two operating modes:
* Generating patterns from modified template
* Generating patterns programmatically

### 1. Generating patterns from modified template
In this mode, the program accepts a modfied template file and generates patterns based on the template. The template file is a bmp file with usable area of DMD marked in black/white and the surrounding marked in red, which labels the real space pixel status.

The template file can be modified to add different patterns on the usable area of DMD. The program will generate patterns based on the modified template file. The generated patterns in the DMD coordinate space will be saved as a bmp file.

To generate pattern with a modified template, open a terminal window and run 
```
python3 main.py
```
use the pop-up window to select modified template file, and the generated patterns will be saved in the same directory as the template file

### 2. Generating patterns programmatically
There are a few pre-defined patterns written in [utils.py](utils.py). Demo code is in this [Jupyter notebook](DMD_pattern_generation.ipynb).