from PIL import Image
import os
import numpy as np

BLOCKS_WITH_COLORS = {
    'dirt':(80,240,15),
    'crafting bench':(3,240,28)
}

class GameWorld2d:
    def __init__(self,image_dir,name,c1,c2,spawn_pos=(0,0)):
        '''
        c1 and c2 are coordinates of the top left and bottom right corners respectively
        '''

        files = os.listdir(image_dir)
        filtered = []
        for f in files:
            if name in f:
                filtered.append(f)
        '''
        img = frame.crop((w_adj,h_adj,WIDTH-w_adj,HEIGHT-h_adj))
        img = np.reshape(np.array(list(img.getdata())),(1,SIZE_TENSOR[0],SIZE_TENSOR[1],SIZE_TENSOR[2]))
        '''

        width  = c2[0]-c1[0]
        height = c2[1]-c1[1]

        self.layers = []
        for f in filtered:
            img = Image.open(image_dir + f).crop((c1[0],c1[1],c2[0],c2[1]))
            img.show()
            self.layers.append(np.reshape(np.array(list(img.getdata())),(width,height,4)))

    self.grid = []
    convertToBlocks()
    self.pos = spawn_pos

    def findClosest(self,obj,number):
        '''
        returns the nearest >number< occurances of >obj<
        how to select nearest instance of target object?
            1) closest euclidian

            2) for each n closest euclidian, calculate distance with A* then pick actual shortest path
        '''
        locs = []
        for col in range(0,len(self.grid)):
            for row in range(0,len(self.grid[col])):
                item = self.grid[col][row]
                if item == obj:
                    locs.append((col,row))
                    if len(locs) == number:
                        return locs
        return locs

    def convertToBlocks(self):
        '''
        coverts to blocks
        '''
        return False
    def makeMovement(self,pos):
        if not validMovement(pos):
            return False
        self.pos = pos
    def validMovement(self,pos):
        return self.grid[pos[0]][pos[1]] == None

GameWorld2d('resources/2D/','train1',(1220,400),(1400,550))
