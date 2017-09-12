class GameObject:
    def __init__(self):
        do = 'nothing'
    def __eq__(self,other):
        return False
    def __ne__(self,other):
        return False
class Resource(GameObject):
    def __init__(self,tags):
        self.tags = tags
    def __eq__(self,other):
        return False
    def __ne__(self,other):
        return False
