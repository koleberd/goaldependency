from PIL import Image
import os
import numpy as np
import time
import imageio
import random

#for parsing input world
BLOCK_IND = {
    (0,0,0):'wall',
    (127,127,127):'stone',
    (255,0,0):'iron ore',
    (237,28,36):'iron ore',
    (255,201,14):'coal',
    (185,122,87):'wood',
    (34,177,76):'crafting bench',
    (255,255,255):None
}
#for printing output world
COLOR_IND = {
    'wood': (0,0,255),
    'crafting bench':(0,255,0),
    'iron ore':(255,150,100),
    'stone':(130,130,130),
    'coal':(70,70,70),
    'default':(0,0,0)
}

BLOCK_TYPES = ['wood','crafting bench', 'iron ore', 'stone', 'coal']


class GameWorld2d:
    def __init__(self,image_path,spawn_pos=(0,0),spawn_random=False):

        img = Image.open(image_path)
        self.width = img.width
        self.height = img.height
        self.grid = [ [None for y in range(self.height)] for x in range(self.width)]
        for col in range(self.width):
            for row in range(self.height):
                self.grid[col][row] = parseBlock(img.getpixel((col,row)))
        self.pos = spawn_pos
        if spawn_random:
            self.randomizePos()
        self.yaw = 0

    def randomizePos(self):
        npos = (random.randint(0,self.width-1),random.randint(0,self.height-1))
        while self.grid[npos[0]][npos[1]] != None:
            npos = (random.randint(0,self.width-1),random.randint(0,self.height-1))
        self.pos = npos
        return npos

    def findClosest(self,obj,number):#not used
        '''
        finds the closest <number> instances of <obj> from self.pos, in terms of euclidian distance
        '''
        locs = []
        for col in range(0,self.width):
            for row in range(0,self.height):
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

    def printWorld(self,path=[]):
        render = Image.new('RGB',(self.width,self.height),color=(255,255,255))
        for col in range(0,self.width):
            for row in range(0,self.height):
                if self.grid[col][row] != None:
                    block =  self.grid[col][row]
                    if block in COLOR_IND.keys():
                        render.putpixel((col,row),COLOR_IND[block])
                    else:
                        render.putpixel((col,row),(255,0,0))
        for pos in path:
            render.putpixel(pos,(193,28,181))

        render.show()

    def saveWorld(self,path=[],name=time.time(),resize=4):
        render = Image.new('RGB',(self.width,self.height),color=(255,255,255))

        for col in range(0,self.width):
            for row in range(0,self.height):
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
        if resize != 1:
            render = resize_no_blur(render,resize)
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
        self.grid[pos[0]][pos[1]] = newVal

    def renderPath(self,path):
        render = Image.new('RGB',(self.width,self.height),color=(255,255,255))

        for col in range(0,self.width):
            for row in range(0,self.height):
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
        return render

    def rayCast(self,angle,distance):
        ng = (self.yaw+angle+360)%360
        angr = np.deg2rad(ng)
        xmul = 1 if ng <= 90 or ng > 270 else -1
        ymul = -1 if ng > 90 and ng <= 270 else 1
        #print(xmul,ymul)
        xpos = self.pos[0]
        ypos = self.pos[1]
        xdis = xpos + (xmul * distance)
        ydis = ypos + (ymul * distance)
        xmax = self.width
        ymax = self.height

        xstart = max([min([xpos,xdis]),0])
        ystart = max([min([ypos,ydis]),0])
        xend = min([max([xpos+1,xdis]),xmax])
        yend = min([max([ypos+1,ydis]),ymax])

        #print(xstart,ystart,xend,yend)

        grd_cp = [[False for x in range(0,self.height)] for y in range(0,self.width)]

        for col in range(xstart,xend):
            for row in range(ystart,yend):
                if self.grid[col][row] != None:
                    lcor_y = xmul * ymul * .5 + row
                    lcor_x = col - .5
                    rcor_y = xmul * ymul * -.5 + row
                    rcor_x = col + .5
                    #print(lcor_x,lcor_y,rcor_x,rcor_y)
                    angs = np.arctan2([lcor_x-xpos+.5,rcor_x-xpos+.5],[lcor_y-ypos+.5,rcor_y-ypos+.5])
                    #print(angs)
                    l_ang = angs[0]
                    r_ang = angs[1]
                    if (angr > l_ang and angr < r_ang and xmul == 1) or (angr < l_ang and angr > r_ang and xmul != 1):
                        grd_cp[col][row] = True
        cl_x = -1
        cl_y = -1
        cl_d = -1
        for col in range(0,len(grd_cp)):
            for row in range(0,len(grd_cp[0])):
                if grd_cp[col][row]:
                    dist = distance_between(self.pos,[col,row])
                    if cl_d == -1 or dist < cl_d:
                        cl_x = col
                        cl_y = row
                        cl_d = dist


        return cl_d, self.grid[cl_x][cl_y]
        #find actual distance to intersection with location
        #return distance and the object #and coord tuple

    def getKernel(self,radius):
        '''
        startx = max([0,self.pos[0]-radius])
        starty = max([0,self.pos[1]-radius])
        endx = min([self.width-1,self.pos[0]+radius])
        endy = min([self.height-1,self.pos[1]+radius])
        '''

        bl_ind = {None:0,'wood':1,'stone':2,'crafting bench':3,'iron ore':4,'coal':5,'wall':6}
        res = [[0 for x in range(0,radius*2+1)] for y in range(0,radius*2+1)]
        for col in range(self.pos[0]-radius,self.pos[0]+radius+1):
            for row in range(self.pos[1]-radius,self.pos[1]+radius+1):
                if not (row < 0 or col < 0 or col >= self.width or row >= self.height):
                    res[col+radius-self.pos[0]][row+radius-self.pos[1]] = bl_ind[self.grid[col][row]]
        return np.array(res).flatten()
        #return [[bl_ind[self.grid[i][j]] for j in range(0,self.height)] for i in range(0,self.width)]

    def getAverageDistances(self):
        print('Calculating average distances...')
        dist_set = {}
        for bl in BLOCK_TYPES:
            dist_set[bl] = []
        for bl in BLOCK_TYPES:
            print('Calculating for ' + bl)
            for col in range(0,self.width):
                for row in range(0,self.height):
                    if self.grid[col][row] == bl:
                        for col2 in range(0,self.width):
                            for row2 in range(0,self.height):
                                if self.grid[col2][row2] == None:
                                    dist_set[bl].append(distance_between((col,row),(col2,row2)))
        avg_set = {}
        for bl in BLOCK_TYPES:
            avg_set[bl] = sum(dist_set[bl])/float(len(dist_set[bl]))
        return avg_set

    def getDensities(self):
        counts = [0 for x in range(len(BLOCK_TYPES)+1)]
        total = 0
        for col in range(self.width):
            for row in range(self.height):
                if self.grid[col][row] in BLOCK_TYPES:
                    counts[BLOCK_TYPES.index(self.grid[col][row])] += 1
                    total += 1
                elif self.grid[col][row] != None:
                    counts[-1] += 1
                    total += 1

        counts = [float(x)/float(self.width*self.height) for x in counts]
        print(BLOCK_TYPES)
        print(counts)
        print('total: ' + str(float(total)/float(self.width*self.height)))
        return counts


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
    npx = (pixel[0],pixel[1],pixel[2])
    for match in BLOCK_IND:
        if match == npx:
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
