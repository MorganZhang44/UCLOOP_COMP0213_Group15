from object.object import object

class cube(object):
    def __init__(self):
        super().__init__(0.05,"./urdf/cube_small.urdf")
        
    def reset(self):
        super().reset()