import json
from os import listdir
import time
#from os.path import isfile,join


###
#manages the player's inventory. has some lasting functionality from when this was used in minecraft
###

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
