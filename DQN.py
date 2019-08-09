from jqdatasdk import *
import glo
import os
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras.optimizers import *
import StockSimEnv as Env
import random

# tf预设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽通知信息和警告信息
# DQN超参数
train_times = 100000
train_step = 1000
gamma = 0.6
mini_batch_size = 15
target_net_update_frequency = 100
sample_frequency = 4
epsilon = 0.05
experience_pool_size = 20
money_limit = 10 ^ 5
# 全局变量
experience_pool = [None for i in range(experience_pool_size)]
current_net = None
target_net = None
experience_cursor = 0
# 初始化
auth("13074581737", "trustno1")
glo.__init__()


class Experience:
    stock_state = [[]]
    agent_state = []
    action = 0
    reward = 0
    stock_state2 = [[]]
    agent_state2 = []

    def __init__(self, stock_state, agent_state, action, reward, stock_state2, agent_state2):
        self.stock_state = stock_state
        self.agent_state = agent_state
        self.action = action
        self.reward = reward
        self.stock_state2 = stock_state2
        self.agent_state2 = agent_state2

    def get_experience(self):
        return {'stock_state': self.stock_state, 'agent_state': self.action, 'action': self.action,
                'reward': self.reward, 'stock_state2': self.stock_state2, 'agent_state2': self.agent_state2}


def get_state():
    """获取当前环境状态"""
    return Env.get_state()


def dqn_loss(y_true, y_pred):
    """
    DQN神经网络所需的损失函数
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))


def build_model():
    # 搭建全连接神经网
    """
    输入state
    输出action(>0买入，<0卖出)money（交易额）
    输出价值Q
    """

    x_stock_state = Input(shape=(6, 20), name="x_stock_state")
    x_agent_state = Input(shape=(3,), name="x_agent_state")
    x_stock_state01 = Conv1D(filters=1, kernel_size=4, padding='same')(x_stock_state)
    merge_stock_state = Flatten()(x_stock_state01)
    merge = Concatenate()([merge_stock_state, x_agent_state])
    dense01 = Dense(16, activation='relu')(merge)
    dense02 = Dense(16, activation='tanh')(dense01)
    dense03 = Dense(8, activation='softmax')(dense02)
    # index=0表示action，大于0为买入多少钱的股票，小于0为卖出
    # index=1表示Q
    # TODO：尚未想好如何保证输出的是最大的Q和action
    # 期望输出最大Q
    output = Dense(2, name='output')(dense03)
    model = Model(inputs=[x_stock_state, x_agent_state], outputs=[output])
    model.compile(optimizer=Adagrad(), loss=dqn_loss)
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


def build_target_net():
    """
    构建一个同主网络相同的网络作为TargetNN
    """
    return build_model()


def update_target_net(main_network):
    """
    持久化主网络，再读取并返回，间接实现深拷贝
    """
    main_network.save('main_net.h5')
    return load_model('main_net.h5', custom_objects={'dqn_loss': dqn_loss})


def append_experience(experience):
    """
    向经验池中添加经验，如果经验池已满，则从index=0开始覆盖
    :param experience: 要添加的经验
    :return: None
    """
    global experience_cursor
    experience_cursor = (experience_cursor + 1) % experience_pool_size
    experience_pool[experience_cursor] = experience


def observe_env():
    """
    观察环境信息，向环境输入各种action进行尝试，获取环境返回的state和reward
    将获得的Experience写入experience_pool
    """
    for i in range(experience_pool_size):
        # 获取状态
        # TODO：随机交易日
        obs_stock_state, obs_agent_state = get_state()
        # 随机生成action
        a = random.randrange(-money_limit, money_limit + 1)
        execute_action(a)
        env_stock_state, env_agent_state, reward = Env.get_state_with_reward(a)
        experience = Experience(obs_stock_state, obs_agent_state, a, reward, env_stock_state, env_agent_state)
        append_experience(experience)


def get_experience_batch():
    """
    获取经验训练包
    :return: 从经验池中随机采样的经验训练包
    """
    return random.sample(experience_pool, mini_batch_size)


def execute_action(action):
    """
    执行action
    :param action: 动作
    :return: None
    """
    if action > 0:
        Env.buy_in(glo.get_value("stockCode"), action)
    elif action < 0:
        Env.sell_out(glo.get_value("stockCode"), action)


def get_info_from_experience_list(experience_list):
    """
    从经验列表中整理状态动作和奖励
    :param experience_list: 经验列表
    :return: list：股票状态、智能体状态、动作、奖励、next_股票状态、next_智能体状态
    """
    stock_res = []
    agent_res = []
    action_res = []
    reward_res = []
    stock_res2 = []
    agent_res2 = []
    for ex in experience_list:
        stock_res.append(ex.stock_state)
        agent_res.append(ex.agent_state)
        action_res.append(ex.action)
        reward_res.append(ex.reward)
        stock_res2.append(ex.stock_state2)
        agent_res2.append(ex.agent_state2)
    action_res = np.array(action_res).reshape(mini_batch_size, 1).tolist()
    reward_res = np.array(reward_res).reshape(mini_batch_size, 1).tolist()
    return stock_res, agent_res, action_res, reward_res, stock_res2, agent_res2


# 训练
def train_network():
    global current_net
    global target_net
    # TODO:直接读取预训练神经网
    current_net = build_model()
    target_net = update_target_net(current_net)
    print('神经网初始化完成')
    update_counter = 0
    # 初始化经验池
    observe_env()
    print('经验池初始化完成')
    for episode in range(train_times):
        # 初始化状态
        start_stock_state, start_agent_state = get_state()
        print('状态初始化完成')
        for i in range(train_step):
            # 由当前状态预测action
            temp = current_net.predict([[start_stock_state], [start_agent_state]])
            temp_action = temp[0][0]
            # epsilon选取action
            ep = random.randrange(1, 100)
            if ep <= epsilon * 100:
                # 随机选取action
                action = random.randrange(-money_limit, money_limit + 1)
                print('epsilon选取action' + str(action))
            else:
                # 通过网络输出的Qmax选择action
                action = temp_action
                print('神经网选取action' + str(action))
            # 在环境中执行actions
            execute_action(action)
            # 获取此时环境的状态
            env_stock_state, env_agent_state, reward = Env.get_state_with_reward(action)
            # 生成经验
            experience = Experience(start_stock_state, start_agent_state, action, reward, env_stock_state,
                                    env_agent_state)
            # 保存经验
            append_experience(experience)
            # 更新状态
            start_stock_state = env_stock_state
            start_agent_state = env_agent_state
            # 从经验池中随机采样
            sample = random.sample(experience_pool, mini_batch_size)
            # 从样本中整理状态和动作list
            s_stock, s_agent, s_action, s_reward, s_stock2, s_agent2 = get_info_from_experience_list(sample)
            # 使用target_net预测Q_target
            Q_target = np.array(s_reward) + gamma * target_net.predict([s_stock2, s_agent2])[:, 1].reshape(
                mini_batch_size, 1)
            train_target = np.append(Q_target, np.array(s_action), axis=1)
            # 执行训练过程
            current_net.fit([s_stock, s_agent], train_target, batch_size=mini_batch_size)
            print('完成current_net训练')
            # 记录次数
            update_counter += 1
            # 更新TargetNet
            if update_counter >= target_net_update_frequency:
                print('更新target_net')
                update_counter = 0
                target_net = update_target_net(current_net)
    print('DQN训练完成，保存模型')
    target_net.save('target_net.h5')


train_network()
