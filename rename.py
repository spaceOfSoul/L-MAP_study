import os
import glob

path = './2022_01_energy/2022_01/'

files = glob.glob(path + '/*')

for f in files:
    os.rename(f,os.path.join(path, os.path.basename(f)[6:]))