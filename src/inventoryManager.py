import json
import thirdparty.nbt as nbt
from os import listdir
import time
#from os.path import isfile,join


CNTR_DST = 36
C_ST = 824
R_ST = 553
IC_C_ST = 1004
IC_R_ST = 422
ICR_C = 1116
ICR_R = 441
CB_C_ST = 868
CB_R_ST = 420
CBR_C = 1056
CBR_R = 457
HOTBAR_JUMP = 44

class InventoryManager:
    def __init__(self):
        with open('json/inventoryItem.json') as rsjs:
            self.resourceIndex = json.load(rsjs)
        self.inventory = []
        for x in range(0,4):
            self.inventory.append([])#rows
            for y in range(0,9):
                self.inventory[x].append(('empty',0))
    def depositStack(self,obj,qnt):
        rem = qnt
        for row in range(0,4):
            if rem == 0:
                break
            for col in range(0,9):
                if rem == 0:
                    break
                slot = self.inventory[row][col]
                if obj == slot[0] and slot[1] < self.resourceIndex[obj]['stackSize']:
                    amt = self.resourceIndex[obj]['stackSize'] - slot[1]
                    if amt > rem:
                        amt = rem
                    self.inventory[row][col] = (obj,amt+slot[1])
                    rem -= amt
                elif slot[0] == 'empty':
                    amt = self.resourceIndex[obj]['stackSize'] - slot[1]
                    if amt > rem:
                        amt = rem
                    self.inventory[row][col] = (obj,amt)
                    rem -= amt
    def deposit(self,obj,qnt):
        rem = qnt
        for row in range(0,4):
            if rem == 0:
                break
            for col in range(0,9):
                if rem == 0:
                    break
                slot = self.inventory[row][col]
                if slot[0] == 'empty':
                    amt = self.resourceIndex[obj]['stackSize'] - slot[1]
                    if amt > rem:
                        amt = rem
                    self.inventory[row][col] = (obj,amt)
                    rem -= amt
    def withdraw(self,obj,qnt):
        rem = qnt
        for row in range(3,-1,-1):
            if rem == 0:
                break
            for col in range(8,-1,-1):
                if rem == 0:
                    break
                slot = self.inventory[row][col]
                if obj == slot[0] and slot[1] > 0:
                    amt = slot[1]
                    if amt > rem:
                        amt = rem
                    self.inventory[row][col] = (obj,slot[1]-amt)
                    rem -= amt
                if self.inventory[row][col][1] == 0:
                    self.inventory[row][col] = ('empty',0)
    def abandon(self):
        return False
    def swap(self,r1,c1,r2,c2):
        hold = self.inventory[r1][c1]
        self.inventory[r1][c1] = self.inventory[r2][c2]
        self.inventory[r2][c2] = hold

    def invCoordOf(self,obj):
        for row in range(3,-1,-1):
            for col in range(8,-1,-1):
                if self.inventory[row][col][0] == obj:
                    return (row,col)
        return None
    def coordOf(self,obj):
        for row in range(3,-1,-1):
            for col in range(8,-1,-1):
                if self.inventory[row][col][0] == obj:
                    return self.coordSlot(row,col)
        return None

    def coordSlot(self,r,c):

        resc = C_ST + CNTR_DST*c
        resr = R_ST + CNTR_DST*r
        if r == 0:
            resr = R_ST + 2*CNTR_DST + HOTBAR_JUMP
        return (resc,resr)

    def coordInvC(self,x,y):#coordinates of inventory craft
        #if x or y are 3, then returns the coordinates of the output of the crafting bench
        r = int(y)
        c = int(x)
        if r == 3 or c == 3:
            return (ICR_C,ICR_R)
        return (IC_C_ST + c * CNTR_DST, IC_R_ST + r * CNTR_DST)

    def coordCbC(self,x,y):#coordinates of crafting bench craft
        #if x or y are 3, then returns the coordinates of the output of the crafting bench
        r = int(y)
        c = int(x)
        if r == 3 or c == 3:
            return (CBR_C,CBR_R)
        return (CB_C_ST + c * CNTR_DST, CB_R_ST + r * CNTR_DST)
    def coordNextEmpty(self):
        for row in range(0,4):
            for col in range(0,9):
                if self.inventory[row][col][0] == 'empty':
                    return self.coordSlot(row,col)
    def translate(self,obj):
        name = obj.split(':')[1]
        translate = {
            'planks':'wood plank',
            'log':'wood'
        }
        if name in translate.keys():
            name = translate[name]
        return name
    def parseInventory(self):
        #return False

        parsedInv = [None]*36
        pInv = InventoryManager()
        folder = 'C:\\Users\\Kirevikyn\\AppData\\Roaming\\.minecraft\\saves\\TEST_ENV_1\\playerdata'
        filename = folder + '\\' + listdir(folder)[0]
        #print(filename)
        playerfile = nbt.NBTFile(filename,'rb')
        items = playerfile['Inventory'].tags
        for item in items:
            #parsedInv[item['Slot'].value]={self.translate(item['id'].value):item['Count'].value}
            pInv.inventory[int(item['Slot'].value/9)][item['Slot'].value%9] = (self.translate(item['id'].value),item['Count'].value)
        return pInv
    def __eq__(self,other):
        for row in range(0,4):
            for col in range(0,9):
                if self.inventory[row][col][0] != other.inventory[row][col][0] and self.inventory[row][col][1] != other.inventory[row][col][1]:
                    return False
        return True

    def __ne__(self,other):
        return not (self == other)
#print()
#print(InventoryManager().inventory)
