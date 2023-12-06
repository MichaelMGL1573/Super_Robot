from SteeringBehaviors import Wander, Seek
import SimulationEnvironment as sim
from DQN import DQN
import os as os
import pickle
import numpy as np
import torch
import numpy.linalg as la
import csv
from collections import deque
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

def get_network_param(sim_env, action, distance, scaler):
    sensor_readings = sim_env.raycasting()
    is_near_goal = (distance < 55)
    # network_param = np.append(sensor_readings, [action, sim_env.robot.body.position.x, sim_env.robot.body.position.y]) #unutilized 0 added to match shape of scaler
    network_param = np.append(sensor_readings, [action, is_near_goal]) #unutilized 0 added to match shape of scaler
    # network_param = scaler.transform(network_param.reshape(1,-1))
    # network_param = network_param.flatten()[:-1]
    network_param = torch.FloatTensor(network_param)  # torch.float32
    return network_param

def goal_seeking(goals_to_reach):
    sim_env = sim.SimulationEnvironment()
    action_repeat = 50
    # steering_behavior = Wander(action_repeat)
    steering_behavior = Seek(sim_env.goal_body.position)

    action_space = np.arange(-10,11)
    
    state_size = sim_env.raycasting().shape[0] + 2
    action_size = len(action_space)
    dqn_agent = DQN(state_size, action_size)
    #load model
    # model = DQN()
    if os.path.exists('saved/saved_model.pkl'):
        dqn_agent = torch.load('saved/saved_model.pkl')
    
    dqn_agent.last_memory_total = 0
    dqn_agent.cur_loss_total = 0
    dqn_agent.last_avg_loss = 9999
    # dqn_agent.memory_total = dqn_agent.memory_counter
    #load normalization parameters
    # scaler = pickle.load(open("saved/scaler.pkl", "rb"))

    # dqn_agent.model.eval()
    accurate_predictions, false_positives, missed_collisions = 0, 0, 0
    robot_turned_around = False
    actions_checked = []
    collisions_total = 0
    goals_reached = 0
    collision_list = np.empty((0, 7))
    collision_pos_list = np.empty((0, 4))
    # pos_collision_probability = np.empty((0,15))
    pcp = 0
    # postGoalReachedFactor = 1
    # postGoalReachedActionCount = 0
    # if os.path.exists('saved/collisionlist.csv'):
    #     collision_list = np.genfromtxt('saved/collisionlist.csv', delimiter=',')
    
    # if os.path.exists('saved/collisionPosList.csv'):
    #     collision_pos_list = np.genfromtxt('saved/collisionPosList.csv', delimiter=',')        

    writer = SummaryWriter('cifar-10')
    network_params = deque(np.empty((0, 6+9)), maxlen=10) # np.empty((0, 6))
    seek_vector = sim_env.goal_body.position - sim_env.robot.body.position
    last_seek_vector = la.norm(seek_vector)
    done = 0
    reward = 0
    step_total = 0
    round_step_total = 0
    round_step_max = 5000
    reward_total = 0
    while goals_reached < goals_to_reach:

        seek_vector = sim_env.goal_body.position - sim_env.robot.body.position
        if la.norm(seek_vector) < 50:
            sim_env.move_goal()
            steering_behavior.update_goal(sim_env.goal_body.position)
            # sim_env.move_robot()
            goals_reached += 1
            round_step_total = 0
            # reward_total = 0
            # network_params[-1][4] = 1
            # network_params[-1][2] = 5
            # postGoalReachedFactor = 1
            # postGoalReachedActionCount = 0
            # sim_env.turn_robot_around()  #To avoid collision, because RL can't learn how to turn around.
            continue
        # # action_space = np.arange(-5,6)
        # actions_available = []
        # preds = []
        # for action in action_space:
        #     network_param = get_network_param(sim_env, action, scaler)
        #     prediction = dqn_agent.model(network_param)
        #     preds.append(round(prediction.item()*10000)/10000)
        #     # print(f"action: {action}, prediction: {prediction}", end="\r\n")
        #     if prediction.item() < .15:  # collision probability < .25?
        #         actions_available.append([action, prediction.item()])

        # # pos_collision_probability = np.append(pos_collision_probability, [[sim_env.robot.body.position.angle_degrees, sim_env.robot.body.position.angle, sim_env.robot.body.position.x,sim_env.robot.body.position.y,preds[10], preds[9], preds[8], preds[7], preds[6], preds[5], preds[4], preds[3], preds[2], preds[1], preds[0]]], axis=0)
        # if len(actions_available) == 0:
        #     # sim_env.turn_robot_around()
        #     # robot_turned_around = True
        #     # continue
        #     if robot_turned_around is False:
        #         sim_env.turn_robot_around()
        #         robot_turned_around = True
        #         continue
        #     else:
        #         actions_available = [[-8, 0]]
        # else:
        #     robot_turned_around = False
        round_step_total += 1
        step_total += 1
        if step_total > 25000:
            step_total = 25000
            
        if round_step_total>round_step_max:
            round_step_total = round_step_max
        action, _ = steering_behavior.get_action(sim_env.robot.body.position, sim_env.robot.body.angle)
        # min, closest_action, closest_pred = 9999, 9999, 0
        # for a, a_pred in actions_available:
        #     diff = abs(action - a)
        #     if diff < min:
        #         min = diff
        #         closest_action = a
        #         closest_pred = a_pred

        state = get_network_param(sim_env, action, la.norm(seek_vector), 0)  # state = next_state
        preds, closest_action_index = dqn_agent.act(state)
        preds = F.softmax(torch.from_numpy(preds), dtype=float)
        preds = preds.numpy()
        closest_action = action_space[closest_action_index]
        closest_pred = preds[closest_action_index]
        action_diff = abs(action - closest_action)
        # if closest_pred < 0.05:
        #     if robot_turned_around is False:
        #         sim_env.turn_robot_around()
        #         robot_turned_around = True
        #         continue
        #     else:
        #         closest_action = -8
        # else:
        #     robot_turned_around = False
                
        steering_force = steering_behavior.get_steering_force(closest_action, sim_env.robot.body.angle)
        for action_timestep in range(action_repeat):
            if action_timestep == 0:
                _, collision, sensor_readings = sim_env.step(preds, steering_force, False, closest_pred)
            else:
                _, collision, _ = sim_env.step(preds, steering_force, False, closest_pred)
            
            
            seek_vector = sim_env.goal_body.position - sim_env.robot.body.position            
            next_state = get_network_param(sim_env, action, la.norm(seek_vector) ,0)
            
            if collision:
                # sim_env.add_collision()
                collision_pos_list = np.append(collision_pos_list, [[sim_env.robot.body.position.angle_degrees, sim_env.robot.body.position.angle, sim_env.robot.body.position.x,sim_env.robot.body.position.y]], axis=0)
                # if action_timestep < action_repeat * .3: #in case prior action caused collision
                #     network_params[-1][-1] = collision #share collision result with prior action
                # if(len(network_params)>0):
                    # network_params[-1][2] = -1  # reward = -1
                # collision_list = np.append(collision_list, network_params, axis=0)
                collisions_total = collisions_total + 1  
                break

        # next_state = get_network_param(sim_env, action, 0)

        sim_env.write_caption(f'collisions: {collisions_total}, goals_reached: {goals_reached}, [action : {action:.4f}]  [{closest_action} : {closest_pred:.4f}]')       
        if collision:
            reward = -200 * (1 + dqn_agent.learn_step_counter/5000)
            done = 1
            network_params[-1][2] = reward
            network_params[-1][4] = 1
            steering_behavior.reset_action()
        else:
            seek_vector = sim_env.goal_body.position - sim_env.robot.body.position
            
            round_reward_steps_factor = 1 - round_step_total/round_step_max
            # https://blog.csdn.net/abcdefg90876/article/details/108459475
            if la.norm(seek_vector) < 50:
                reward = 1000 #* round_reward_steps_factor
                # done = 1
                # break
            else:
                # if last_seek_vector - la.norm(seek_vector) > 0:
                #     reward = 1
                # else:
                #     reward = -1
                factor = (la.norm(seek_vector)-50)
                if factor <= 1:
                    factor = 1
                reward_distance_factor = (1+5/factor)
                reward_closing = ((last_seek_vector - la.norm(seek_vector)))#/10
                action_diff_factor = (1 + 1/(1+action_diff))
                reward = reward_closing * action_diff_factor  # reward_distance_factor  #* round_reward_steps_factor
                last_seek_vector = la.norm(seek_vector)
                done = 0
                            
        if len(network_params) == 10:
            if network_params[4] == 1:
                dqn_agent.add_collision(network_params[0])
            else:
                dqn_agent.store_transition(network_params[0])

        network_params.append([state, closest_action_index, reward, next_state, done]) #[[sensor_readings[0], sensor_readings[1], sensor_readings[2], sensor_readings[3], sensor_readings[4]]]            
        done = 0
        reward_total += reward
        
        writer.add_scalar("reward/step Total", reward_total, dqn_agent.learn_step_counter)
        # dqn_agent.memory_maxsize = 10000
        # if dqn_agent.memory_counter > 100:
        dqn_agent.replay(writer)
        
        if( dqn_agent.memory_total > 0 and dqn_agent.memory_total % 100 == 0):
            torch.save(dqn_agent, 'saved/saved_model.pkl') # save model and tensor  
            
        if (goals_reached > 0):
            accurate_predictions = 1 - collisions_total/goals_reached
            false_positives = collisions_total/goals_reached
            missed_collisions = goals_reached - collisions_total
        print(f'dqn_agent.memory_total: {dqn_agent.memory_total}, dqn_agent.epsilon:{dqn_agent.epsilon}', end="\r", flush=True)
        
        # pcp = pcp + 1
        # if (pcp > 1000):
        #     torch.save(pos_collision_probability, "saved/pos_collision_probability.dd")
        #     pcp = 0      
                
    
    
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    goals_to_reach = 20000
    goal_seeking(goals_to_reach)    
