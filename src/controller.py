import pyautogui
import time
from PIL import ImageGrab
from PIL import Image
from playerState import *
from gameState import *

defaultColors = [
    (0,0,0), #coal
    (255,0,0), #crafting bench
    (0,0,255), #diamond ore
    (0,255,0), #stone
    (128,128,128), #furnace
    (255,128,0), #iron ore
    (128,64,0), #wood
    (255,255,255) #bedrock
]

resourcesWithColors = {
    'coal ore':(0,0,0), #coal
    'crafting bench':(255,0,0), #crafting bench
    'diamond ore':(0,0,255), #diamond ore
    'stone':(0,255,0), #stone
    'furnace':(128,128,128), #furnace
    'iron ore':(255,128,0), #iron ore
    'wood':(128,64,0), #wood
    'bedrock':(255,255,255) #bedrock
}

UNID = '?'

def moveForward(dur):
    pyautogui.keyDown('w')
    time.sleep(dur)
    pyautogui.keyUp('w')

def hitBlock(dur):
    pyautogui.moveRel(0,100)
    pyautogui.mouseDown()
    time.sleep(dur)
    pyautogui.mouseUp()

def normPix(pixel,normalize):

    return (norm(pixel[0],normalize),norm(pixel[1],normalize),norm(pixel[2],normalize))
def norm(num,amt):
    #255 -> 255
    # 224 v
    #192 -> 192
    #160 v
    #128 -> 128
    #92 v
    #64  -> 64
    #32 v
    #0   -> 0
    val = int(amt*round((num-1)/amt+.0000001))
    if val == 256:
        val = 255
    return val

def filt(pixel):
    adj = 32
    pix = (0,0,0)
    r = 0
    g = 0
    b = 0
    if pixel[0] != 0:
        r += adj
    if pixel[1] != 0:
        g += adj
    if pixel[2] != 0:
        b += adj
    return (pixel[0] + r, pixel[1] + g, pixel[2] + b)

#the AI part, just returns a normalized sum for counts now.
def scaleCounts(obj):
    total = 0;
    for x in obj:
        if x != UNID:
            total += obj[x]
    res = {}
    for x in obj:
        if x != UNID:
            res[x] = 1 - round(obj[x]/total,2)
    return res

def matchWithResource(pixel,table):
    normPixel = normPix(filt(pixel),64)
    #print(normPixel)
    ret = UNID
    for item in table:
        if normPixel == table[item]:
            ret = item
    return ret




def getFOV():
    RESIZE_FACTOR = 16
    startT = time.time()
    frame = ImageGrab.grab()
    frame = frame.resize((int(frame.width/RESIZE_FACTOR),int(frame.height/RESIZE_FACTOR)))#.quantize(colors=16)

    clipWidth = .9 #9/10 width
    clipHeight = .1 #1/10 height

    startX = int((frame.width - (frame.width * clipWidth)) / 2)
    startY = int((frame.height - (frame.height * clipHeight)) / 2)
    width = int(frame.width * clipWidth)
    height = int(frame.height * clipHeight)
    #frame.show()
    data = []
    for i in range(0,width):
        data.append([])
        for j in range(0,height):
            data[i].append(matchWithResource(frame.getpixel((startX+i,startY+j)),resourcesWithColors))

    #print('process frame: ' + str(time.time() - startT))
    #print(width)
    #print(height)
    #print('--------------')
    objTally = {}
    for x in data:
        for y in x:
            if y not in objTally.keys():
                objTally[y] = 1
            else:
                objTally[y] += 1
    return scaleCounts(objTally)
    #print(frame.height)
    #print(frame.width)
    #print(frame.getpixel((5,5)))

def getLookedAt():
    return None
def runSim():
    time.sleep(3)
    #moveForward(5)
    #getFOV()
    for x in range(0,10):
        time.sleep(1)
        getFOV()

def getCurrentGameState():
    ps = PlayerState(lookedAt=getLookedAt())
    return GameState(ps=ps,fov=getFOV())
