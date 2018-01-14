from PIL import Image
import os
import numpy as np
import time
'''
GameWorld2d is used for simulating a 2d world based on a top down rendering of a minecraft world.
    It can take multiple layers of a world but compresses them down into one layer. The top block takes precedence over lower blocks.
'''




BLOCK_IND_DARK = {
    (0,31,0,255):None,
    (29,8,0,255):'wood',
    (43,0,0,255):'wood',
    (59,4,0,255):'crafting bench',
    (45,12,0,255):'iron ore',
    (25,0,0,255):'iron ore',
    (0,41,50,255):'diamond ore',
    (15,15,15,255):'stone',
    (0,0,0,255):'stone'
}
BLOCK_IND = {
    (155,122,99,255):'iron ore',
    (175,142,118,255):'iron ore',
    (90,69,36,255):'wood',
    (102,81,47,255):'wood',
    (59,59,59,255):'coal',
    (69,69,69,255):'coal',
    (100,100,100,255):'stone',
    (116,116,116,255):'stone',
    (86,157,66,255):None,
    (65,136,45,255):None,
    (160,105,59,255):'crafting bench'
}

COLOR_IND = {
    'wood': (0,0,255),
    'crafting bench':(0,255,0),
    'iron ore':(255,150,100),
    'stone':(130,130,130),
    'coal':(70,70,70),
    'default':(0,0,0)
}


class GameWorld2D:
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
            layer = [ [None for y in range(height)] for x in range(width)]
            #img.show()
            #self.layers.append(np.reshape(np.array(list(img.getdata())),(width,height,4)))
            for col in range(width):
                for row in range(height):
                    layer[col][row] = parseBlock(img.getpixel((col,row)))
            self.layers.append(layer)


        #flatten

        self.grid = [ [None for y in range(height)] for x in range(width)]
        for layer in self.layers:
            for col in range(width):
                for row in range(height):
                    #block = self.parseBlock(layer[col][row])
                    block = layer[col][row]
                    if block != None:
                        self.grid[col][row] = block



        self.pos = spawn_pos

    def findClosest(self,obj,number):
        '''
        finds the closest <number> instancse of <obj> from self.pos, in terms of euclidian distance
        '''
        locs = []
        for col in range(0,len(self.grid)):
            for row in range(0,len(self.grid[col])):
                item = self.grid[col][row]
                if item == obj:
                    locs.append((col,row))
        orgLen = len(locs)
        dists = {}
        for loc in locs:
            dists[loc] = distance_between(self.pos,loc)
        ranked = []
        while len(ranked) < number and len(ranked) < orgLen:
            minP = 0
            minV = dists[locs[0]]
            for i in range(len(locs)):
                loc = locs[i]
                if dists[loc] < minV:
                    minV = dists[loc]
                    minK = loc
                    minP = i
            ranked.append(locs[minP])
            del locs[i]
        return ranked

    def isExposed(self,col,row):
        top = row - 1 >= 0 and self.grid[col][row-1] == None
        left = col - 1 >= 0 and self.grid[col-1][row] == None
        right = col + 1 < len(self.grid) and self.grid[col+1][row] == None
        bottom = row + 1 < len(self.grid[0]) and self.grid[col][row+1] == None
        return top or left or right or bottom

    def findClosestLayered(self,obj,number):
        '''
        finds the closest <number> instancse of <obj> from self.pos, in terms of euclidian distance. bottom object is always closer than upper objects in layers.
        '''
        locs = []
        for col in range(0,len(self.layers[0])):
            for row in range(0,len(self.layers[0][0])):
                for layer in range(0,len(self.layers)):
                    item = self.layers[layer][col][row]
                    if item == obj and self.isExposed(col,row):
                        locs.append((layer,col,row))
        orgLen = len(locs)
        dists = {}
        for loc in locs:
            dists[loc] = distance_between(self.pos,(loc[1],loc[2]))
        ranked = []
        while len(ranked) < number and len(ranked) < orgLen:
            minP = 0
            minV = dists[locs[0]]
            for i in range(len(locs)):
                loc = locs[i]
                if dists[loc] < minV:
                    minV = dists[loc]
                    minK = loc
                    minP = i
            ranked.append(locs[minP])
            del locs[i]
        return ranked

    def printWorld(self,path=[]):
        render = Image.new('RGB',(len(self.grid),len(self.grid[0])),color=(255,255,255))
        for col in range(0,len(self.grid)):
            for row in range(0,len(self.grid[0])):
                if self.grid[col][row] != None:
                    block =  self.grid[col][row]
                    if block in COLOR_IND.keys():
                        render.putpixel((col,row),COLOR_IND[block])
                    else:
                        render.putpixel((col,row),(255,0,0))
        for pos in path:
            render.putpixel(pos,(193,28,181))

        render.show()

    def saveWorld(self,path=[],name=time.time()):
        render = Image.new('RGB',(len(self.grid),len(self.grid[0])),color=(255,255,255))

        for col in range(0,len(self.grid)):
            for row in range(0,len(self.grid[0])):
                if self.grid[col][row] != None:
                    block =  self.grid[col][row]
                    if block in COLOR_IND.keys():
                        render.putpixel((col,row),COLOR_IND[block])
                    else:
                        render.putpixel((col,row),COLOR_IND['default'])

        path_dark = len(path)
        for pos in path:
            mul = path_dark/len(path) + .3
            render.putpixel(pos,((int(193*mul),int(28*mul),int(181*mul))))
            path_dark -= 1
        render = resize_no_blur(render,10)
        render.save('simulation/2Dpath/'+str(name)+'.jpg')

    def astar(self,start,goal):
        start_time = time.time()
        evaluatedNodes = []
        discoveredNodes = [start]
        bestReachedFrom = {}
        fromStartCost = {}
        fromStartToGoalThroughNode = {}
        for col in range(0,len(self.grid)):
            for row in range(0,len(self.grid[0])):
                fromStartCost[(col,row)] = 2**16
                fromStartToGoalThroughNode[(col,row)] = 2**16
        fromStartCost[start] = 0
        fromStartToGoalThroughNode[start] = estimate_cost(start,goal)

        while len(discoveredNodes) != 0:
            if time.time() - start_time > 5.0:
                print('A* timeout')
                return None
            #print(len(evaluatedNodes))
            current = getNodeWithLowestCost(discoveredNodes,fromStartToGoalThroughNode)
            if current == goal:
                #print('A* finished in ' + str(time.time()-start_time) + ' seconds.')
                return reconstruct_path(bestReachedFrom,current)
            discoveredNodes = deleteNodeFromList(discoveredNodes,current)
            evaluatedNodes.append(current)
            neighbor_set = self.getNeighbors(current)
            if isAdjacentTo(current,goal):
                #neighbor_set.append(goal)
                #print('A* finished in ' + str(time.time()-start_time) + ' seconds.')
                return reconstruct_path(bestReachedFrom,current)
            for neighbor in neighbor_set:
                if neighbor in evaluatedNodes:
                    continue
                if neighbor not in discoveredNodes:
                    discoveredNodes.append(neighbor)
                tentativeFromStartcost = fromStartCost[current] + distance_between(current,neighbor)
                if tentativeFromStartcost >= fromStartCost[neighbor]:
                    continue
                bestReachedFrom[neighbor] = current
                fromStartCost[neighbor] = tentativeFromStartcost
                fromStartToGoalThroughNode[neighbor] = fromStartCost[neighbor] + estimate_cost(neighbor,goal)
        return None

    def getNeighbors(self,pos):
        neighbors = []
        if pos[0] >= 1 and self.isValidMovement((pos[0]-1,pos[1])):#left
            neighbors.append((pos[0]-1,pos[1]))
        if pos[0] < len(self.grid) - 2 and self.isValidMovement((pos[0]+1,pos[1])):#right
            neighbors.append((pos[0]+1,pos[1]))
        if pos[1] >= 1 and self.isValidMovement((pos[0],pos[1]-1)):#up
            neighbors.append((pos[0],pos[1]-1))
        if pos[1] < len(self.grid[0]) - 2 and self.isValidMovement((pos[0],pos[1]+1)):#down
            neighbors.append((pos[0],pos[1]+1))
        return neighbors

    def isValidMovement(self,pos,exception='not none'):
        return self.grid[pos[0]][pos[1]] == None or self.grid[pos[0]][pos[1]] == exception

    def updateLoc(self,pos,newVal):
        self.layers[pos[0]][pos[1]][pos[2]] = newVal
        emptyFlat = False
        curr = None
        for layer in range(0,len(self.layers)):
            this_layer = self.layers[layer][pos[1]][pos[2]]
            if curr == None and this_layer != None:
                curr = this_layer
        self.grid[pos[1]][pos[2]] = curr




def resize_no_blur(img,factor):
    res = Image.new('RGB',(img.width*factor,img.height*factor))
    for col in range(0,img.width):
        for row in range(0,img.height):
            for rcol in range(col*factor,(col+1)*factor):
                for rrow in range(row*factor,(row+1)*factor):
                    res.putpixel((rcol,rrow),img.getpixel((col,row)))
    return res




def parseBlock(pixel):
    #print(type(pixel))
    for match in BLOCK_IND:
        if match == pixel:
            return BLOCK_IND[match]
    return 'OCCUPIED'

def isAdjacentTo(p1,p2):
    #print(p1,p2,abs(p2[0]-p1[0]) == 1 != abs(p2[1]-p1[1]) == 1)
    return (np.abs(p2[0]-p1[0]) == 1 and np.abs(p2[1]-p1[1]) == 0 ) or (np.abs(p2[0]-p1[0]) == 0 and np.abs(p2[1]-p1[1]) == 1)

def distance_between(n1,n2):
    return np.sqrt(np.abs(n2[0]-n1[0])**2 + np.abs(n2[1]-n1[1])**2)

def estimate_cost(start,goal):
    #return np.sqrt(np.abs(goal[0]-start[0])**2 + np.abs(goal[1]-start[1])**2)
    return distance_between(start,goal)

def reconstruct_path(movements,target):
    total_path = [target]
    while target in movements.keys():
        target = movements[target]
        total_path.append(target)
    return total_path

def getNodeWithLowestCost(vals,costs):
    lowest = costs[vals[0]]
    lowestN = None
    for val in vals:
        if costs[val] <= lowest:
            lowest = costs[val]
            lowestN = val
    return lowestN

def deleteNodeFromList(vals,node):
    for x in range(len(vals)):
        if vals[x] == node:
            del vals[x]
            return vals

#GameWorld2d('resources/2D/','train1',(1230,410),(1370,520))
#GameWorld2d('resources/2D/','train2',(528,454),(528+46,454+46))
#wrld = GameWorld2D('resources/2D/','train4',(552,391),(552+42,391+42),spawn_pos=(2,2))
#wrld.printWorld()
