import pybullet as p
import pybullet_data
import time
import numpy as np
from gripper.pawl_2f import pawl_2f
from gripper.pawl_3f import pawl_3f
from object.cube import cube
from object.cylinder import cylinder
import pandas as pd
from ML.Classifier import ClassifierGraspPlanner


class SimEnv:
    def __init__(self,robot,object):
        self.cid = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1,
            cameraYaw=40,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.2]
        )
        p.loadURDF("plane.urdf")
        if robot=="2f":
            self.pawl = pawl_2f()
        else:
            self.pawl = pawl_3f(object)
        if object == "cube":
            self.obj = cube()
        else:
            self.obj = cylinder()
        
        
    def catch(self):
        # Open gripper
        self.pawl.open_gripper()
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)

        # Lower slightly (1 cm)
        randposition = self.pawl.get_randpos(self.obj.height)
        self.pawl.move_gripper(randposition[0],p.getQuaternionFromEuler(randposition[1]))
        
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)
        
        near_pos = [randposition[0][i]*self.pawl.ratio for i in range(2)]
        near_pos.append((randposition[0][2]-self.obj.height/2)*(self.pawl.ratio+0.05)+self.obj.height/2)
        self.pawl.move_gripper(near_pos,p.getQuaternionFromEuler(randposition[1]),force=1100)
        
        for _ in range(30):
            p.stepSimulation()
            time.sleep(1./240.)

        # Close gripper to grasp
        self.pawl.close_gripper()
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)

        # Lift cube
        self.pawl.move_gripper([0,0,0.3],p.getQuaternionFromEuler(randposition[1]),force=500)
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)

        # Move along x to drop location
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1./240.)
            
        pos, orn = p.getBasePositionAndOrientation(self.obj.cube_id)
        if pos[2]>0.1:
            return randposition,1
        else:
            return randposition,0

        
    def reset(self):
        self.obj.reset()
        self.pawl.reset()
        
    def get_data(self,num,csv_path):
        df = {
                "x": [],
                "y": [],
                "z": [],
                "roll": [],
                "pitch": [],
                "yaw": [],
                "label": []
                }
        for i in range(num):
            print(f"Generating data {i}")
            self.reset()
            data = self.catch()
            if data[1]==1:
                print("This grasp is Success")
            else:
                print("This grasp is Fail")
            #s = f"{data[0][0][0]} {data[0][0][1]}  {data[0][0][2]}  {data[0][1][0]}  {data[0][1][1]}  {data[0][1][2]}  {data[1]}"
            df["x"].append(data[0][0][0])
            df["y"].append(data[0][0][1])
            df["z"].append(data[0][0][2])
            df["roll"].append(data[0][1][0])
            df["pitch"].append(data[0][1][1])
            df["yaw"].append(data[0][1][2])
            df["label"].append(data[1])
            
        df = pd.DataFrame(df)
        df.to_csv(csv_path,index=False)
        self.finish()
        
    def test(self,num,model_path):
        planner = ClassifierGraspPlanner()
        planner.load(model_path)
        right = 0
        for i in range(num):
            print(f"Testing {i}")
            self.reset()
            data = self.catch()
            input = np.array([data[0][0][0],data[0][0][1],data[0][0][2],data[0][1][0],data[0][1][1],data[0][1][2]])
            input = input.reshape(1,-1)
            result = planner.predict(input)
            print(f"Prediction: {result[0]}, Actual: {data[1]}",end=' ')
            if result[0]==data[1]:
                print("Correct")
                right+=1
            else:
                print("Wrong")
        print(f"Accuracy: {right/num}")
        self.finish()
                
            
            
        
    def finish(self):
        p.disconnect()
        

