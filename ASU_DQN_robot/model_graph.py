from SteeringBehaviors import Wander, Seek
import SimulationEnvironment as sim
# from Networks import Action_Conditioned_FF
import os as os

import pickle
import numpy as np
import torch
import numpy.linalg as la
import csv

def goal_seeking(goals_to_reach):
    sim_env = sim.SimulationEnvironment()
    action_repeat = 100
    # steering_behavior = Wander(action_repeat)
    steering_behavior = Seek(sim_env.goal_body.position)


    # steering_force = steering_behavior.get_steering_force(0, sim_env.robot.body.angle)
    # sim_env.step(steering_force, False, 0)
    
    pos_collision_list = torch.load("saved/pos_collision_probability.dd")    
    for idx, pc in enumerate(pos_collision_list):
        sim_env.add_track((pc[2],pc[3]))
        # sim_env.add_sensors_probability(pc[0], pc[2:4], pc[4:], sensor_range=150.0)
        # if idx > 50 : 
        #     break
        
    collision_pos_list = np.genfromtxt('saved/collisionPosList.csv', delimiter=',')   
    for idx, pc in enumerate(collision_pos_list):
        sim_env.add_collision((pc[2],pc[3]))

    steering_force = steering_behavior.get_steering_force(0, sim_env.robot.body.angle)
    sim_env.step([0,0,0,0,0,0,0,0,0,0,0], steering_force, False, 0)
    
    aa = input("Finish the simulation, Yes or No? Y/N ")
            
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    goals_to_reach = 2000
    goal_seeking(goals_to_reach)
