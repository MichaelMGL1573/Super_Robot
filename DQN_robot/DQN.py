import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.nn.functional as F
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# https://blog.csdn.net/weixin_44732379/article/details/127821138

# # 定义Q网络
# class QNetwork(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(QNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_size, 64, dtype=float)
#         self.fc2 = nn.Linear(64, 64, dtype=float)
#         self.fc3 = nn.Linear(64, action_size, dtype=float)
#         self.fc1.weight.data.normal_(0, 0.1)    # 初始化权重，用二值分布来随机生成参数的值
#         self.fc2.weight.data.normal_(0, 0.1)
#         self.fc3.weight.data.normal_(0, 0.1)
        
#     def forward(self, state):
#         global map
#         map = state
#         map = copy.deepcopy(map.detach())
#         x = torch.relu(self.fc1(map))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)


# 深度网络，全连接层
class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(QNetwork, self).__init__()
        # n_states状态个数
        self.fc1 = nn.Linear(n_states, 300)
        self.fcm = nn.Linear(300, 300)
        # n_actions动作个数
        self.fc2 = nn.Linear(300, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)    # 初始化权重，用二值分布来随机生成参数的值
        self.fc2.weight.data.normal_(0, 0.1)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.3)
        
    # 前向传播
    def forward(self, x):
        # 这里以一个动作为作为观测值进行输入，然后把他们输出给10个神经元
        # x = self.fc1(x)
        # 激活函数
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fcm(x))
        # x = self.drop2(x)
        # 经过10个神经元运算过后的数据， 把每个动作的价值作为输出。
        out = self.fc2(x)
        # out = torch.softmax(self.fc2(x), dim=0)
        return out
    
# 定义DQN算法
class DQN:
    epoch = 0
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # self.model = QNetwork(state_size, action_size)
        self.eval_net, self.target_net = QNetwork(state_size, action_size), QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.scaler = MinMaxScaler()  # default tuple: (0,1)  x' = (x-Xmin)/(Xmax-Xmin)
        self.loss = nn.MSELoss()
        self.last_memory_total = 0
        self.memory_counter = 0
        self.memory_total = 0
        self.memory_maxsize = 5000
        # self.memory = deque(maxlen=self.memory_maxsize)
        self.memory = np.zeros((self.memory_maxsize, 15+16))  # 2*2(state和next_state,每个x,y坐标确定)+2(action和reward),存储2000个记忆体
        self.collision_counter = 0
        self.collision_total = 0
        self.collision_maxsize = 1000
        # self.collision = deque(maxlen=self.collision_maxsize)
        self.collision = np.zeros((self.collision_maxsize, 15+16))  # 2*2(state和next_state,每个x,y坐标确定)+2(action和reward),存储2000个记忆体
        self.learn_step_counter = 0  # target网络学习计数
        self.batch_size = 1000
        self.cur_loss_total = 0
        self.last_avg_loss = 9999
        self.cur_avg_loss = 0
        
    # def add_member(self, member):       
    #     if (member != []):
    #         transition=", transition)
 
    # def remember(self, state, action, reward, next_state, done):        
    #     self.memory_count += 1
    #     self.memory.append([state, action, reward, next_state, done])


    # def remember(self, state, action, reward, next_state, done):        
    #     self.memory_count += 1
    #     self.memory.append([state, action, reward, next_state, done])

    # def sample_memory(self, batch_size):
    #     #  eval net是 每次learn 就进行更新
    #     #  更新逻辑就是从记忆库中随机抽取BATCH_SIZE个（32个）数据。
    #     # 使用记忆库中批量数据
    #     # sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 从 数据库中 随机 抽取 BATCH_SIZE条数据
    #     sample_batch_size = batch_size #int(self.batch_size*(self.memory_counter/(self.memory_counter+self.collision_counter)))
    #     # sample_index = np.random.choice(self.memory_counter, sample_batch_size)  # 200个中随机抽取16个作为batch_size
    #     if (sample_batch_size > len(self.memory)):
    #         memory_batch = random.sample(self.memory, len(self.memory))
    #     else:
    #         memory_batch = random.sample(self.memory, sample_batch_size)
    #     # use the collision data.
    #     # Cause few collision data will be added into memory when training is well.
    #     # Model will be overfit and loss will be increased when collision data is disapeared in the train set.
    #     # solve the 自举（Bootstrapping） problem
    #     collision_batch_size = batch_size/10 
    #     #int(self.batch_size*(self.collision_counter/(self.memory_counter+self.collision_counter)))        
    #     if (collision_batch_size > len(self.collision)):
    #         collition_batch = random.sample(self.collision, len(self.collision))
    #     else:
    #         collition_batch = random.sample(self.collision, collision_batch_size)
    #     # collision_sample_index = np.random.choice(self.collision_counter, collision_batch_size)  # 200个中随机抽取16个作为batch_size
    #     # collision_sample_index = np.random.choice(self.collision_counter, self.collision_counter)  # 200个中随机抽取16个作为batch_size
    #     # sample_index= [ 34  48 153  60   5 140  74  81  93  85 138  33 118  90  11 124]
    #     # print("<learn> sample_index=", sample_index)
    #     # non_collision_memory = self.memory[sample_index, :]  #  抽取BATCH_SIZE个（16个）个记忆单元， 把这BATCH_SIZE个（16个）数据打包.
    #     # collision_memory = self.collision[collision_sample_index, :]  #  抽取BATCH_SIZE个（16个）个记忆单元， 把这BATCH_SIZE个（16个）数据打包.
    #     if len(memory_batch) == 0:
    #         memory = collition_batch
    #     elif len(collition_batch) == 0:
    #         memory = memory_batch
    #     else:
    #         # memory = memory_batch
    #         memory = memory_batch + collition_batch
    #         # memory.append(collition_batch)
            
    #         # np.append(non_collision_memory, collision_memory, axis=0)
               
    #     states, actions, rewards, next_states, dones = list(zip(*memory))
    #     states = [list(row) for row in states]
    #     actions = [list(row) for row in actions]
    #     rewards = [list(row) for row in rewards]
    #     next_states = [list(row) for row in next_states]
    #     dones = [list(row) for row in dones]
        
    #     return torch.FloatTensor(np.array(list(states))), torch.LongTensor(np.array(list(actions))), torch.FloatTensor(np.array(list(rewards))), torch.FloatTensor(np.array(list(next_states))), torch.LongTensor(np.array(list(dones)))    
    
    # # 存储数据
    # # 本次状态，执行的动作，获得的奖励分， 完成动作后产生的下一个状态。
    # # 存储这四个值
    # # def store_transition(self, state, action, reward, next_state, done):
    # def store_transition(self, member): 
    #     self.memory.append(member)
    #     self.memory_counter += 1
    #     self.memory_total += 1
    #     if self.memory_counter >= self.memory_maxsize:
    #         self.memory_counter = self.memory_maxsize

    # # 存储数据
    # # 本次状态，执行的动作，获得的奖励分， 完成动作后产生的下一个状态。
    # # 存储这四个值
    # # def store_transition(self, state, action, reward, next_state, done):
    # def add_collision(self, member):         
    #     self.collision.append(member)
    #     self.collision_counter += 1
    #     self.collision_total += 1
    #     if self.collision_counter >= self.collision_maxsize:
    #         self.collision_counter = self.collision_maxsize
            
    # 存储数据
    # 本次状态，执行的动作，获得的奖励分， 完成动作后产生的下一个状态。
    # 存储这四个值
    # def store_transition(self, state, action, reward, next_state, done):
    def store_transition(self, member): 
        # 把所有的记忆捆在一起，以 np类型
        # 把 三个矩阵 s ,[a,r] ,s_  平铺在一行 [a,r]是因为 他们都是 int 没有 [] 就无法平铺 ，并不代表把他们捆在一起了
        #  np.vstack()是把矩阵进行列连接
        # transition = np.hstack((state, [action, reward], next_state, done))
        transition = np.hstack((member))
        # state= [0.25 0.  ] action= 3 reward= 1 next_state= [0. 0.]
        # <store_transition> transition= [0.25 0.   3.   1.   0.   0.  ]
        #print("<store_transition>tensor(action, dtype=float)
 
        # index 是 这一次录入的数据在 MEMORY_CAPACITY 的哪一个位置
        index = self.memory_total % self.memory_maxsize  # 满了就覆盖旧的
        # 如果，记忆超过上线，我们重新索引。即覆盖老的记忆。
        self.memory[index, :] = transition  # 将transition添加为memory的一行
        #print("<store_transition> memory=", self.memory)
        self.memory_counter += 1
        self.memory_total += 1
        if self.memory_counter >= self.memory_maxsize:
            self.memory_counter = self.memory_maxsize

    # 存储数据
    # 本次状态，执行的动作，获得的奖励分， 完成动作后产生的下一个状态。
    # 存储这四个值
    # def store_transition(self, state, action, reward, next_state, done):
    def add_collision(self, member): 
        # 把所有的记忆捆在一起，以 np类型
        # 把 三个矩阵 s ,[a,r] ,s_  平铺在一行 [a,r]是因为 他们都是 int 没有 [] 就无法平铺 ，并不代表把他们捆在一起了
        #  np.vstack()是把矩阵进行列连接
        # transition = np.hstack((state, [action, reward], next_state, done))
        transition = np.hstack((member))
        # state= [0.25 0.  ] action= 3 reward= 1 next_state= [0. 0.]
        # <store_transition> transition= [0.25 0.   3.   1.   0.   0.  ]
        #print("<store_transition> transition=", transition)
 
        # index 是 这一次录入的数据在 MEMORY_CAPACITY 的哪一个位置
        index = self.collision_total % self.collision_maxsize  # 满了就覆盖旧的
        # 如果，记忆超过上线，我们重新索引。即覆盖老的记忆。
        self.collision[index, :] = transition  # 将transition添加为memory的一行
        #print("<store_transition> memory=", self.memory)
        self.collision_counter += 1
        self.collision_total += 1
        if self.collision_counter >= self.collision_maxsize:
            self.collision_counter = self.collision_maxsize
            
    def act(self, state):
        if np.random.rand() <= self.epsilon:  #E_greedy stratege
            return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), random.randrange(11)+5   #random.randrange(self.action_size)
        act_values = self.eval_net(state)
        # output = F.log_softmax(act_values) #, dim=1
        return act_values.detach().numpy(), torch.argmax(act_values) # pi* = argmaxQ*(s,a)

    def replay(self, writer):

        # if self.memory_counter < 10 or self.collision_total < 1:
        #     return

        # if self.memory_counter < self.memory_maxsize:  # use this when collecting data in very beginning.
        if self.memory_counter < self.batch_size:
            return

        if self.memory_total - self.last_memory_total < 10:            
            return
        else:
            self.last_memory_total = self.memory_total
            
        # print("<learn>")
        # target net 更新频率,用于预测，不会及时更新参数
        # target parameter update  是否要更新现实网络
        # target Q现实网络 要间隔多少步跟新一下。 如果learn步数 达到 TARGET_REPLACE_ITER  就进行一次更新
        if self.learn_step_counter % 100 == 0:
            # 把最新的eval 预测网络 推 给target Q现实网络
            # 也就是变成，还未变化的eval网
            self.cur_avg_loss = self.cur_loss_total / 100
            #if (self.cur_avg_loss < self.last_avg_loss):
            self.target_net.load_state_dict((self.eval_net.state_dict()))
            
            self.last_avg_loss = self.cur_avg_loss
            self.cur_loss_total = 0
            
            writer.add_scalar("Mean Eval Loss/100", self.cur_avg_loss, self.learn_step_counter)
            # 'fc1.weight', 'fc1.bias', 'fc2.weight', ....
            #print("<learn> eval_net.state_dict()=", (self.eval_net.state_dict()))
        self.learn_step_counter += 1
        
        #  eval net是 每次learn 就进行更新
        #  更新逻辑就是从记忆库中随机抽取BATCH_SIZE个（32个）数据。
        # 使用记忆库中批量数据
        # sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 从 数据库中 随机 抽取 BATCH_SIZE条数据
        sample_batch_size = self.batch_size #int(self.batch_size*(self.memory_counter/(self.memory_counter+self.collision_counter)))
        if (sample_batch_size > self.memory_counter):
            sample_index = np.random.choice(self.memory_counter, self.memory_counter)  # 200个中随机抽取16个作为batch_size
        else:
            sample_index = np.random.choice(self.memory_counter, sample_batch_size)  # 200个中随机抽取16个作为batch_size
        
        # use the collision data.
        # Cause few collision data will be added into memory when training is well.
        # Model will be overfit and loss will be increased when collision data is disapeared in the train set.
        # solve the 自举（Bootstrapping） problem
        collision_batch_size = 100 #int(self.batch_size*(self.collision_counter/(self.memory_counter+self.collision_counter)))
        if (collision_batch_size > self.collision_counter):
            collision_sample_index = np.random.choice(self.collision_counter, self.collision_counter)  # 200个中随机抽取16个作为batch_size
        else:
            collision_sample_index = np.random.choice(self.collision_counter, collision_batch_size)  # 200个中随机抽取16个作为batch_size
        # collision_sample_index = np.random.choice(self.collision_counter, self.collision_counter)  # 200个中随机抽取16个作为batch_size
        # sample_index= [ 34  48 153  60   5 140  74  81  93  85 138  33 118  90  11 124]
        # print("<learn> sample_index=", sample_index)
        non_collision_memory = self.memory[sample_index, :]  #  抽取BATCH_SIZE个（16个）个记忆单元， 把这BATCH_SIZE个（16个）数据打包.
        collision_memory = self.collision[collision_sample_index, :]  #  抽取BATCH_SIZE个（16个）个记忆单元， 把这BATCH_SIZE个（16个）数据打包.
        if len(non_collision_memory)==0:
            memory = collision_memory
        elif len(collision_memory)==0:
            memory = non_collision_memory
        else:
            memory = np.append(non_collision_memory, collision_memory, axis=0)
        
        # state, action, reward, next_state, done = self.sample_memory(self.batch_size)
        
        # memory= [[-0.5  -0.25  1.    0.   -0.5   0.  ]
        #          [-0.5   0.25  1.    0.   -0.5   0.25]
        #          [ 0.    0.25  2.    0.    0.25  0.25]
        #  ...]
        # print("<learn> memory=", memory)
 
        state = torch.FloatTensor(memory[:, :14])    # # 32个记忆的包，包里是（当时的状态） 所有行里取0,1
        # state = self.scaler.fit_transform(state)
        
        # 下面这些变量是 32个数据打包的变量
        #   state= tensor([[-0.5000, -0.2500],
        #         [-0.5000,  0.2500],
        #         [ 0.0000,  0.2500],
        #   ...]
        # print("<learn> state=", state)
        action = torch.LongTensor(memory[:, 14:15])   # # 32个记忆的包，包里是（当时做出的动作）2
        #   action= tensor([[1],
        #         [1],
        #         [2],
        #   ...]
        # print("<learn> action=", action)
        reward = torch.FloatTensor(memory[:, 15:16])   # # 32个记忆的包，包里是 （当初获得的奖励）3
        next_state = torch.FloatTensor(memory[:, 16:30])  # 32个记忆的包，包里是 （执行动作后，下一个动作的状态）4,5ction = torch.LongTensor(memory[:, 6:7])
        done = torch.LongTensor(memory[:, 30:31])
        # next_state = self.scaler.fit_transform(next_state)
        # q_eval w.r.t the action in experience
        # q_eval的学习过程
        # self.eval_net(state).gather(1, action)  输入我们包（32条）中的所有状态 并得到（32条）所有状态的所有动作价值，
        # .gather(1,action) 只取这32个状态中 的 每一个状态的最大值
        # 预期价值计算 ==  随机32条数据中的最大值
        # 计算loss,
        # q_eval:所采取动作的预测value,
        # q_target:所采取动作的实际value
        # a.gather(0, b)分为3个部分，a是需要被提取元素的矩阵，0代表的是提取的维度为0，b是提取元素的索引。
        # 当前状态的预测：
        # 输入现在的状态state，通过forward()生成所有动作的价值，根据价值选取动作，把它的价值赋值给q_eval
        q_eval = self.eval_net(state).gather(1, action) # eval_net->(64,4)->按照action索引提取出q_value
 
        #  state= tensor([[-0.2500, -0.2500],
        #         [-0.2500, -0.2500],
        #         [-0.5000, -0.5000],
        # ...]
        # eval_net(state)= tensor([[-0.1895, -0.2704, -0.3506, -0.3678],
        #         [-0.1895, -0.2704, -0.3506, -0.3678],
        #         [-0.2065, -0.2666, -0.3501, -0.3738],
        # ...]
        #  action= tensor([[0],
        #         [1],
        #         [0],
        # ...]
        # q_eval= tensor([[-0.1895],
        #         [-0.2704],
        #         [-0.2065],
        # ...]
        # print("<learn> eval_net(state)=", self.eval_net(state), "q_eval=", q_eval)
 
        # 下一步状态的预测：
        # 计算最大价值的动作：输入下一个状态 进入我们的现实网络 输出下一个动作的价值  .detach() 阻止网络反向传递，我们的target需要自己定义该如何更新，它的更新在learn那一步
        # 把target网络中下一步的状态对应的价值赋值给q_next；此处有时会反向传播更新target，但此处不需更新，故加.detach()
        q_next = self.target_net(next_state).detach()   # detach from graph, don't backpropagate
        # 计算对于的最大价值
        # q_target 实际价值的计算  ==  当前价值 + GAMMA（未来价值递减参数） * 未来的价值
        # max函数返回索引加最大值，索引是1最大值是0 torch.max->[values=[],indices=[]] max(1)[0]->values=[]
        aa = q_next.max(1)[0].unsqueeze(1)
        q_target = reward + 0.9 * aa * (1 - done)# label # shape (batch, 1)
        #   q_next= tensor([[-0.1943, -0.2676, -0.3566, -0.3752],
        #                   [-0.1848, -0.2731, -0.3446, -0.3604],
        #                   [-0.2065, -0.2666, -0.3501, -0.3738],
        #                   ...]
        #   q_next.max(1)= torch.return_types.max(values=tensor([-0.1943, -0.1848, -0.2065,...]), indices=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        #   q_next.max(1)[0]= tensor([-0.1943, -0.1848, -0.2065,....])
        #   q_next.max(1)[0].unsqueeze(1)= tensor([[-0.1943], [-0.1848], [-0.2065],...])
        #   q_target= tensor([[-0.1749],
        #         [-0.1663],
        #         [-0.1859],
        #   ...]
        # print("<learn> q_target=", q_target, "q_next=", q_next, "q_next.max(1)=", q_next.max(1), "q_next.max(1)[0]=", q_next.max(1)[0], "q_next.max(1)[0].unsqueeze(1)=", q_next.max(1)[0].unsqueeze(1))
        # 通过预测值与真实值计算损失 q_eval预测值， q_target真实值
        
        # target_action = torch.argmax(q_next, dim=1).unsqueeze(1)
        # action = torch.as_ self.memory_count += 1
    #         self.memory.append(member)
        
        # target_action = torch.as_tensor(target_action, dtype=float)
        # loss = self.loss(action, target_action)
        
        loss = self.loss(q_eval, q_target)
        self.cur_loss_total += loss.item()
        
        # self.cost.append(loss.detach().numpy())
        # 根据误差，去优化我们eval网, 因为这是eval的优化器
        # 反向传递误差，进行参数更新
        self.optimizer.zero_grad()  # 梯度重置
        loss.backward()  # 反向求导
        self.optimizer.step()  # 更新模型参数
            
        writer.add_scalar("Train/Loss", loss.item(), self.learn_step_counter)
        if self.epsilon > self.epsilon_min:
            self.epsilon_decay = 0.995
            self.epsilon *= self.epsilon_decay
            # self.epsilon = 1