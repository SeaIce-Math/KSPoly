import sys
import os.path
import pathlib

import warnings
warnings.filterwarnings("ignore")

# Path to all modules used in the project
sys.path.append(str(pathlib.Path(__file__).parent.resolve())+'/lib/')
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))




from .scene import scene
from .frame_data import frame_data
