import pybullet as p

class object:
    def __init__(self,height,urdf_path):
        base_x, base_y = 0, 0
        self.height = height
        cube_start_pos = [base_x, base_y, self.height/2]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.cube_id = p.loadURDF(urdf_path, cube_start_pos, cube_start_orientation)
        
    def reset(self):
        p.resetBasePositionAndOrientation(self.cube_id, [0, 0, self.height/2], [0, 0, 0, 1])
