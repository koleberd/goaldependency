from pynput import keyboard
from pynput.keyboard import Key
from PIL import ImageGrab
from PIL import Image
import os
import json

'''
T stone
I iron
Q wood
N diamond
O gold
F coal
R none
C crafting
U furnace
'''


training_dir = 'training/'
#outfileName = training_dir + 'lookedAtSet.json'
keys_to_classes = {'i':'iron','t':'stone','q':'wood','c':'crafting bench','u':'furnace','n':'diamond','o':'gold','f':'coal','r':'none'}
counter = len(os.listdir(training_dir))-1
prevImg = None
prevImgN = None
def on_press(key):
    global counter
    global prevImg
    global prevImgN
    try:
        if(key.char not in ['w','a','s','d']):
            dumpLastImg()
            prevImgN = keys_to_classes[key.char]
            prevImg = ImageGrab.grab()
        else:
            return
    except AttributeError:#special key pressed
        if(key == Key.esc):
            dumpLastImg()
            printStats()
            #dumpJSON()
            return False
        if key == Key.space and prevImgN != None:
            prevImgN += '_' + 'OOR'
        if key == Key.ctrl:
            prevImgN = None
            prevImg = None
    except KeyError:#not in keys_to_classes, so ignore
        do = 'nothing'

    #print(str(key))
    #mem.append(key)

def on_release(key):
    return

def dumpLastImg():
    global counter
    if prevImg != None:
        counter += 1
        name = training_dir + prevImgN + '_' + str(counter) + '.png'
        prevImg.save(name)
        print('+ ' + name)

def dumpJSON():
    if os.path.isFile(outfileName):
        with open(outfileName) as fl:
            labels.extend(json.load(fl))
    with open(outfileName) as fl:
        json.dump(labels)

def printStats():
    names = [fl.split('.')[0] for fl in os.listdir(training_dir)]
    nameCount = {}

    for fl in names:
        typ = fl.split('_')[0]
        if 'OOR' in fl:
            typ += ' (out of range)'
        if typ in nameCount.keys():
            nameCount[typ] += 1
        else:
            nameCount[typ] = 1

    for item in sorted(nameCount):
        print(str(nameCount[item]) + ' ( ' + str(int(100*nameCount[item]/len(names))) + '% )' + ' - ' + item)
    print(str(len(names)) + ' - total')

with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
    listener.join()
