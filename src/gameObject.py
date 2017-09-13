class GameObject:
    def __init__(self):
        do = 'nothing'
    def __eq__(self,other):#not complete
        return False
    def __ne__(self,other):#not complete
        return False
    def __hash__(self):#not complete
        return 0
class Resource(GameObject):
    def __init__(self,tags):
        self.tags = tags
    def __eq__(self,other):#not complete
        return False
    def __ne__(self,other):#not complete
        return False
    def __hash__(self):#not complete
        return 0
