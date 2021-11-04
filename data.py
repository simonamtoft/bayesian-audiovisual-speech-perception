# Imports
import numpy as np
from glob import glob
from os import path

data_dir = "data"
file_paths = glob(path.join(data_dir, "*.txt"))

# 5 FILES x 7 ROWS  x 5 RESPONSES
# ROW 1: Audio 
# ROW 2: Visual
# ROW 3-7: Visual going from 'b' (row 3) to 'd' (row 7)
# Columns: Audio from 'b' (col 1) to 'd' (col 5)
data = np.array([np.loadtxt(fname) for fname in file_paths])



