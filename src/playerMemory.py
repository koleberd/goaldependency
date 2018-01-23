class PlayerMemory:
    def __init__(self,target=None):
        self.target = target
        self.metrics = {'distance traveled':0}
        self.prev_at = None
        self.curr_at = None
        self.prev_at_parent = None
        self.prev_at_parent_parent = None
        self.prev_at_parent_parent_parent = None
