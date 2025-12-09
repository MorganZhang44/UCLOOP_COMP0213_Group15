from object.object import object

class cylinder(object):
    def __init__(self):
        super().__init__(0.1,"./urdf/cylinder.urdf")
        
    def reset(self):
        super().reset()