###
#contains structures used to track the user's path, world snapshots, and things used for rollbacks as well as extra metrics.
###


class PlayerMemory:
    def __init__(self,target=None):
        self.target = target
        self.metrics = {'distance traveled':0,'path':[]}
        self.prev_at = None
        self.curr_at = None
        self.prev_at_parent = None
        self.prev_at_parent_parent = None
        self.prev_at_parent_parent_parent = None
