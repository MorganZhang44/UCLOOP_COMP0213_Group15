from gripper.Base_pawl import pawls
import pybullet as p
from collections import namedtuple
import math


class pawl_2f(pawls):
    def __init__(self):
        pos = [0,0,0.5]
        quat = p.getQuaternionFromEuler([3.1416,0,0])
        super().__init__("./urdf/2f/2f.urdf", pos, quat)
        
        self.id = p.createConstraint(
            parentBodyUniqueId=self.obj,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0.2]
        )

        self.init_state()
        self.move_gripper(self.base_pos,self.base_quat)
        self.ratio = 0.35
        
        
    def init_state(self):
        num_joints = p.getNumJoints(self.obj)
        JointInfo = namedtuple('JointInfo',['id','name','type','lower','upper','maxForce'])
        self.joints = []
        for i in range(num_joints):
            info = p.getJointInfo(self.obj, i)
            jid = info[0]
            name = info[1].decode()
            jtype = info[2]
            lower = info[8]
            upper = info[9]
            maxForce = info[10]
            self.joints.append(JointInfo(jid,name,jtype,lower,upper,maxForce))
            p.setJointMotorControl2(self.obj,jid,p.VELOCITY_CONTROL,targetVelocity=0,force=0)
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint':1,
                                'left_inner_knuckle_joint':1,
                                'right_inner_knuckle_joint':1,
                                'left_inner_finger_joint':-1,
                                'right_inner_finger_joint':-1}

        self.mimic_parent_id = [j.id for j in self.joints if j.name==mimic_parent_name][0]
        mimic_child_multiplier = {j.id: mimic_children_names[j.name] for j in self.joints if j.name in mimic_children_names}
        for joint_id, multiplier in mimic_child_multiplier.items():
            c = p.createConstraint(self.obj,self.mimic_parent_id,
                                    self.obj,joint_id,
                                    jointType=p.JOINT_GEAR,
                                    jointAxis=[0,1,0],
                                    parentFramePosition=[0,0,0],
                                    childFramePosition=[0,0,0])
            p.changeConstraint(c,gearRatio=-multiplier,maxForce=100,erp=1)
            
        
    def close_gripper(self):
        open_length = 0
        open_angle = 0.715 - math.asin((open_length-0.010)/0.1143)
        p.setJointMotorControl2(self.obj,self.mimic_parent_id,p.POSITION_CONTROL,
                                targetPosition=open_angle, force=30)  # increase force
        return open_angle
    def open_gripper(self):
        open_length = 0.1
        open_angle = 0.715 - math.asin((open_length-0.010)/0.1143)
        p.setJointMotorControl2(self.obj,self.mimic_parent_id,p.POSITION_CONTROL,
                                targetPosition=open_angle, force=60)  # increase force
        return open_angle
        
