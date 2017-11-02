from pynput import keyboard
from pynput.keyboard import Key
from PIL import ImageGrab
from PIL import Image
import os
import json
import time


WIDTH = 1920
HEIGHT = 1080

w_adj = int(WIDTH/4)
h_adj = int(HEIGHT/4)

training_dir = 'training/images/'
output_dir = 'training/proc_images/'
existing = os.listdir(output_dir)
for f in os.listdir(training_dir):
    if f not in existing:
        print('converting ' + f)
        img = Image.open(training_dir +f)
        img = img.crop((w_adj,h_adj,WIDTH-w_adj,HEIGHT-h_adj))
        img.save(output_dir + f)
