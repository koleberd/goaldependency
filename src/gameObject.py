class GameObject:
    def __init__(self,tag):
        self.tags = tags
    def __eq__(self,other):#not complete
        return False
    def __ne__(self,other):#not complete
        return False
    def __hash__(self):#not complete
        return 0
    #returns None if tag doesn't exist, or tag content if it does
    def getTag(self,name):
        return self.tags[name]
class Resource(GameObject):
    def __eq__(self,other):#not complete
        return False
    def __ne__(self,other):#not complete
        return False
    def __hash__(self):#not complete
        return 0
