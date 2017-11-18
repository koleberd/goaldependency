from PIL import Image
import os
import numpy as np
import time

BLOCKS_WITH_COLORS = {
    'dirt':(0,31,0,255),
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
            layer = [ [None for y in range(height)] for x in range(width)]
            #img.show()
            #self.layers.append(np.reshape(np.array(list(img.getdata())),(width,height,4)))
            for col in range(width):
                for row in range(height):
                    layer[col][row] = self.parseBlock(img.getpixel((col,row)))
            self.layers.append(layer)

        self.grid = [ [None for y in range(height)] for x in range(width)]
        for layer in self.layers:
            for col in range(width):
                for row in range(height):
                    #block = self.parseBlock(layer[col][row])
                    block = layer[col][row]
                    if block != None:
                        self.grid[col][row] = block

        self.pos = spawn_pos

        path = self.astar((8,90),(130,50))


        self.printWorld(path)


    def parseBlock(self,pixel):
        #print(type(pixel))
        if tuple(pixel) == BLOCKS_WITH_COLORS['dirt']:
            return None
        return 'OCCUPIED'

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

    def printWorld(self,path=[]):
        render = Image.new('RGB',(len(self.grid),len(self.grid[0])),color=(255,255,255))
        for col in range(0,len(self.grid)):
            for row in range(0,len(self.grid[0])):
                if self.grid[col][row] != None:
                    render.putpixel((col,row),(255,0,0))
        for pos in path:
            render.putpixel(pos,(0,0,0))

        render.show()

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
            #print(len(evaluatedNodes))
            current = getNodeWithLowestCost(discoveredNodes,fromStartToGoalThroughNode)
            if current == goal:
                print('A* finished in ' + str(time.time()-start_time) + ' seconds.')
                return reconstruct_path(bestReachedFrom,current)
            #print(current)
            discoveredNodes = deleteNodeFromList(discoveredNodes,current)
            evaluatedNodes.append(current)
            for neighbor in self.getNeighbors(current):
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

    def isValidMovement(self,pos):
        return self.grid[pos[0]][pos[1]] == None

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


GameWorld2d('resources/2D/','train1',(1230,410),(1370,520))
