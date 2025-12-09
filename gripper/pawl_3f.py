from gripper.Base_pawl import pawls
import pybullet as p


class pawl_3f(pawls):
    GRASP_JOINTS = [1, 4, 7]
    PRESHAPE_JOINTS = [2, 5, 8]
    UPPER_JOINTS = [3, 6, 9]

    def __init__(self,object):
        pos = [0, 0, 0.5]
        quat = p.getQuaternionFromEuler([3.14, 0, 0])
        super().__init__("./urdf/3f/sdh/sdh.urdf", pos, quat)
        self.num_joints = p.getNumJoints(self.obj)
        self.ratio = 0.45
        self.object = object


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


    def open_gripper(self):
        """Gradually open fingers until fully open."""
        for k in range(self.num_joints):
            if k in [2, 5, 8]:
                self._apply_joint_command(k, 0.7)
            elif k in [3, 6, 0] :
                self._apply_joint_command(k, 0.9 if self.object == "cube" else 0)
            elif k in [1, 4, 7] :
                self._apply_joint_command(k, -0.6)

    def _apply_joint_command(self, joint, target):
        p.setJointMotorControl2(self.obj, joint, p.POSITION_CONTROL,
                                targetPosition=target, maxVelocity=10, force=60)

    def get_joint_positions(self):
        return [p.getJointState(self.obj, i)[0] for i in range(self.num_joints)]

    def close_gripper(self):
        for j in [1, 4,7]:
            p.setJointMotorControl2(self.obj, j, p.POSITION_CONTROL,
                                        targetPosition=-0.2 if self.object == "cube" else 0, maxVelocity=10, force=60)
        for k in [2, 5, 8]:
            p.setJointMotorControl2(self.obj, k, p.POSITION_CONTROL,
                                    targetPosition=1, maxVelocity=10, force=100)
