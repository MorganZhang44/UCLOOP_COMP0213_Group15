from algorithm.random_gripper import generate_random_gripper_pose
import pybullet as p


class pawls:
    def __init__(self,urdf_name,pos,quat):
        self.base_pos = pos
        self.base_quat = quat
        self.obj = p.loadURDF(urdf_name, pos, quat, useFixedBase=False)
    def init_state(self):
        pass
    def reset(self):
        p.resetBasePositionAndOrientation(self.obj, self.base_pos, self.base_quat)
        self.move_gripper(self.base_pos,self.base_quat)
    def close_gripper(self):
        pass
    def open_gripper(self):
        pass
    def move_gripper(self,pos,quat,force=1200):
        
        p.changeConstraint(
                self.id,
                jointChildPivot=pos,
                jointChildFrameOrientation=quat,
                maxForce=force
            )
        
    def get_randpos(self,height):
        rand_pose = generate_random_gripper_pose(cube_center=[0,0,height/2],)
        pos = [rand_pose[0],rand_pose[1],rand_pose[2]]
        orn = [rand_pose[3],rand_pose[4],rand_pose[5]]
        return pos,orn