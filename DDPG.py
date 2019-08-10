from time import *
import glo
import os
from keras.optimizers import *
import StockSimEnv as Env
import random
import tensorflow as tf
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import math
import json
from stock_date import *
import plotly as py
import plotly.graph_objs as go
import sys

# f = open('trade.log', 'a')
# sys.stdout = f
# sys.stderr = f
# tf预设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽通知信息和警告信息
# Tensorflow GPU optimization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# DDPG超参数
train_times = 20
train_step = 10000
gamma = 0.99
mini_batch_size = 64
experience_pool_size = 50000
money_limit = glo.ori_money
tau = 0.001
stock_state_size = 6
agent_state_size = 3
action_size = 2
epsilon = 0.3
# 全局变量
experience_pool = [None for i in range(experience_pool_size)]
main_actor_net = None
main_critic_net = None
target_actor_net = None
target_critic_net = None
experience_cursor = 0
date_manager = StockDate()


# auth("13074581737", "trustno1")


class Experience:
    stock_state = [[]]
    agent_state = []
    action = []
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

    def __repr__(self):
        return repr(
            (self.stock_state, self.agent_state, self.action, self.reward, self.stock_state2, self.agent_state2))

    @staticmethod
    def object_hook(d):
        return Experience(d['stock_state'], d['agent_state'], d['action'], d['reward'], d['stock_state2'],
                          d['agent_state2'])


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
    date = StockDate()
    for i in range(experience_pool_size):
        # 每观察train_step轮重置一次环境
        if i % train_step == 0:
            Env.__init__()
        t = random.randint(int(glo.frequency[:-1]), date.date_list.size - 1 - train_step * int(glo.frequency[:-1]))
        date.set_date_with_index(datetime.strptime(date.date_list[t], "%Y-%m-%d %H:%M:%S"), t)
        # 获取状态
        obs_stock_state, obs_agent_state = Env.get_state(date.get_date())
        # 随机生成action
        a = [random.uniform(-1, 1), random.uniform(-1, 1)]
        env_stock_state, env_agent_state, reward = execute_action(a)

        # 处理数据
        obs_stock_state = np.array(obs_stock_state).reshape(1, stock_state_size, glo.count).tolist()
        obs_agent_state = np.array(obs_agent_state).reshape(1, agent_state_size).tolist()
        env_stock_state = np.array(env_stock_state).reshape(1, stock_state_size, glo.count).tolist()
        env_agent_state = np.array(env_agent_state).reshape(1, agent_state_size).tolist()
        a = np.array(a).reshape(1, action_size).tolist()
        reward = np.array(reward).reshape(1, 1).tolist()
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
    :return: 执行动作后的状态和奖励
    """
    return Env.trade(glo.stock_code, action, date_manager.get_date())


def get_info_from_experience_list(experience_list):
    """
    从经验列表中整理状态动作和奖励
    :param experience_list: 经验列表
    :return: list：股票状态、智能体状态、动作、奖励、next_股票状态、next_智能体状态
    """
    stock_res = 0
    agent_res = 0
    action_res = 0
    reward_res = 0
    stock_res2 = 0
    agent_res2 = 0
    for i in range(experience_list.__len__()):
        ex = experience_list[i]
        if i == 0:
            stock_res = np.array(ex.stock_state)
            agent_res = np.array(ex.agent_state)
            action_res = np.array(ex.action)
            reward_res = np.array(ex.reward)
            stock_res2 = np.array(ex.stock_state2)
            agent_res2 = np.array(ex.agent_state2)
        else:
            stock_res = np.concatenate((np.array(ex.stock_state), stock_res), axis=0)
            agent_res = np.concatenate((np.array(ex.agent_state), agent_res), axis=0)
            action_res = np.concatenate((np.array(ex.action), action_res), axis=0)
            reward_res = np.concatenate((np.array(ex.reward), reward_res), axis=0)
            stock_res2 = np.concatenate((np.array(ex.stock_state2), stock_res2), axis=0)
            agent_res2 = np.concatenate((np.array(ex.agent_state2), agent_res2), axis=0)
    return stock_res.tolist(), agent_res.tolist(), action_res.tolist(), reward_res.tolist(), stock_res2.tolist(), agent_res2.tolist()


def run_model(train_model):
    # 初始化
    Env.__init__()
    global main_actor_net
    global main_critic_net
    global target_actor_net
    global target_critic_net
    global epsilon
    global experience_pool
    global date_manager
    actor = ActorNetwork(sess, stock_state_size, agent_state_size, action_size, tau, 0.00001)
    critic = CriticNetwork(sess, stock_state_size, agent_state_size, action_size, tau, 0.000001)
    main_actor_net = actor.model
    target_actor_net = actor.target_model
    main_critic_net = critic.model
    target_critic_net = critic.target_model
    # 绘图数据保存变量
    episode_reward_list = []
    episode_loss_list = []
    if os.path.exists('main_actor_weights.h5'):
        main_actor_net.load_weights('main_actor_weights.h5')
        print("已载入权重main_actor_weights.h5")
    if os.path.exists('main_critic_weights.h5'):
        main_critic_net.load_weights('main_critic_weights.h5')
        print("已载入权重main_critic_weights.h5")
    if os.path.exists('target_actor_weights.h5'):
        target_actor_net.load_weights('target_actor_weights.h5')
        print("已载入权重target_actor_weights.h5")
    if os.path.exists('target_critic_weights.h5'):
        target_critic_net.load_weights('target_critic_weights.h5')
        print("已载入权重target_critic_weights.h5")

    # 如果已经有经验则读取
    if os.path.exists("Data/experience_pool.json"):
        with open("Data/experience_pool.json", "r", encoding="UTF-8") as f:
            s = f.read()
            experience_pool = json.loads(s, object_hook=Experience.object_hook)
        print("已载入经验池")
    # 否则观察环境获取经验
    else:
        observe_env()
        with open("Data/experience_pool.json", "w", encoding='UTF-8') as f:
            json.dump(experience_pool, f, default=lambda obj: obj.__dict__)

    for episode in range(train_times):
        # 重置环境
        Env.__init__()
        profit_list = []
        time_list = []
        quant_list = []
        random_list = []
        reference_list = []
        stock_price_list = []
        candle_list = []
        amount_list = []
        # 初始化状态
        t = random.randint(int(glo.frequency[:-1]),
                           date_manager.date_list.size - 1 - train_step * int(glo.frequency[:-1]))
        date_manager.set_date_with_index(datetime.strptime(date_manager.date_list[t], "%Y-%m-%d %H:%M:%S"), t)
        glo.init_with_oristock(Env.get_stock_price(date_manager.get_date()))

        random_stock = [glo.stock_value[0]]
        random_ori_money = glo.ori_money
        random_money = random_ori_money
        random_amount = glo.stock_value[0][1]

        current_stock_state, current_agent_state = Env.get_state(date=date_manager.get_date())
        current_stock_state = np.array(current_stock_state).reshape(1, stock_state_size, glo.count).tolist()
        current_agent_state = np.array(current_agent_state).reshape(1, agent_state_size).tolist()
        for t in range(train_step):
            # 设置action噪声，增强对环境的探索能力
            a_noise = 0
            if train_model == "train" or train_model == "both":
                a_noise = random.uniform(-epsilon, epsilon)
                # 预防除以0或epsilon=0的情况
                epsilon = epsilon * math.log(train_times - episode*t, train_times) + 0.00000001
            # 假设本轮时间极短，股价不变，本轮内所有股价都从这里获取
            glo.price = Env.get_stock_price(date_manager.get_date())
            stock_price_list.append(glo.price)
            candle_list.append(Env.get_single_stock_state(date=date_manager.get_date()))
            time_list.append(str(date_manager.get_date()))
            print("第" + str(episode + 1) + "轮训练")
            print("     第" + str(t + 1) + "步训练")
            print("日期:" + str(date_manager.get_date()))
            print("生成action")
            # 此处的action不是真正交易的股数，而是[-1,1]，1为当前持有股数
            action = main_actor_net.predict([current_stock_state, current_agent_state])[0]
            print("预测的action:" + str(action))
            print("a_noise:" + str(a_noise))
            action[0] = action[0] + a_noise
            action[1] = action[1] + a_noise
            # 买入过多
            if action[0] > glo.money / glo.price or action[0] > 1:
                action[0] = 1
            if action[0] < -1:
                action[0] = -1
            # 不存在卖出过多
            # 执行动作并获取状态
            print("执行动作" + str(action) + "并获取状态")
            next_stock_state, next_agent_state, reward = execute_action(action)
            step_reward = reward
            print("reward:" + str(reward))
            print("money:" + str(glo.money))
            print("ori_money:" + str(glo.ori_money))
            print("stock_value:" + str(glo.get_stock_total_value(glo.price)))
            print("stock:" + str(glo.stock_value))
            print("stock_amount:" + str(glo.get_stock_amount()))
            amount_list.append(glo.get_stock_amount())
            # 生成经验前预处理
            next_stock_state = np.array(next_stock_state).reshape(1, stock_state_size, glo.count).tolist()
            next_agent_state = np.array(next_agent_state).reshape(1, agent_state_size).tolist()
            action = np.array(action).reshape(1, action_size).tolist()
            reward = np.array(reward).reshape(1, 1).tolist()
            # 生成经验并存储
            experience = Experience(current_stock_state, current_agent_state, action, reward, next_stock_state,
                                    next_agent_state)
            append_experience(experience)

            if train_model == "train" or train_model == "both":
                # 从经验池中采样
                sample = random.sample(experience_pool, mini_batch_size)
                stock_state_, agent_state_, action_, reward_, next_stock_state_, next_agent_state_ = get_info_from_experience_list(
                    sample)
                # 生成yi
                print("生成yi")
                yi = (np.array(reward_) + gamma * np.array(
                    target_critic_net.predict([next_stock_state_, next_agent_state_, target_actor_net.predict(
                        [next_stock_state_, next_agent_state_])]))).tolist()
                # 开始训练
                print("开始训练")
                step_loss = main_critic_net.train_on_batch([stock_state_, agent_state_, action_], [yi])
                a_for_grad = main_actor_net.predict([stock_state_, agent_state_])

                grads = critic.gradients(stock_state_, agent_state_, a_for_grad)
                # print("梯度:"+str(grads))
                actor.train(stock_state_, agent_state_, grads)
                print("learning_rate:" + str(sess.run(actor.learn_rate)))
                actor.update_target()
                critic.update_target()
                episode_reward_list.append(step_reward)
                episode_loss_list.append(step_loss)
                # 每轮结束时画出训练结果图
                if t == train_step - 1:
                    py.offline.plot({
                        "data": [go.Scatter(x=[i for i in range(len(episode_loss_list))], y=episode_loss_list)],
                        "layout": go.Layout(title="episode_loss", xaxis={'title': '步数'}, yaxis={'title': 'loss'})
                    }, auto_open=False, filename='episode_loss.html')
                    py.offline.plot({
                        "data": [go.Scatter(x=[i for i in range(len(episode_reward_list))], y=episode_reward_list)],
                        "layout": go.Layout(title="episode_reward", xaxis={'title': '步数'}, yaxis={'title': 'reward'})
                    }, auto_open=False, filename='episode_reward.html')

            if train_model == "run" or train_model == "both":
                # 画出回测图
                # 现在持有的股票价值+现在的资金
                profit_list.append(
                    (glo.get_stock_total_value(glo.price) + glo.money - glo.ori_money - glo.ori_value) / (
                            glo.ori_value + glo.ori_money))
                # 最开始持有的半仓股票的价值+最开始持有的资金
                reference_list.append(
                    (glo.stock_value[0][1] * glo.price * 100 + glo.ori_money - glo.ori_money - glo.ori_value) / (
                            glo.ori_value + glo.ori_money))
                temp = glo.stock_value[len(glo.stock_value) - 1]
                quant_list.append(temp[1])
                # 随机操作对照组
                random_action = random.uniform(-1, 1)
                random_quant = 0
                if random_action > 0:
                    random_quant = int(random_action * random_money / (100 * glo.price))
                elif random_action < 0:
                    random_quant = int(random_action * random_amount)
                random_amount += random_quant
                random_stock.append([glo.price, random_quant])
                random_money -= glo.price * 100 * random_quant
                random_list.append((glo.price * 100 * random_amount + random_money - glo.ori_money - glo.ori_value) / (
                        glo.ori_money + glo.ori_value))
                f = train_step / 10
                # 训练+绘制回测图模式下调低绘制频率
                path = "sim.html"
                if train_model == "both":
                    f = train_step / 10
                    path = "sim_res/sim_" + str(episode + 1) + ".html"
                if (t + 1) % f == 0 and t != 0:
                    random_scatter = go.Scatter(x=time_list,
                                                y=random_list,
                                                name='随机策略',
                                                line=dict(color='green'),
                                                mode='lines')
                    profit_scatter = go.Scatter(x=time_list,
                                                y=profit_list,
                                                name='DDPG',
                                                line=dict(color='red'),
                                                mode='lines')
                    reference_scatter = go.Scatter(x=time_list,
                                                   y=reference_list,
                                                   name='基准',
                                                   line=dict(color='blue'),
                                                   mode='lines')
                    price_scatter = go.Scatter(x=time_list,
                                               y=stock_price_list,
                                               name='股价',
                                               line=dict(color='orange'),
                                               mode='lines',
                                               xaxis='x',
                                               yaxis='y2',
                                               opacity=0.9)
                    trade_bar = go.Bar(x=time_list,
                                       y=quant_list,
                                       name='交易量（手）',
                                       marker_color='#000099',
                                       xaxis='x',
                                       yaxis='y3',
                                       opacity=0.9)
                    amount_scatter = go.Scatter(x=time_list,
                                                y=amount_list,
                                                name='持股数量（手）',
                                                line=dict(color='rgba(0,204,255,0.6)'),
                                                mode='lines',
                                                fill='tozeroy',
                                                fillcolor='rgba(0,204,255,0.3)',
                                                xaxis='x',
                                                yaxis='y4',
                                                opacity=0.6)
                    """
                    cl = np.array(candle_list)
                    price_candle = go.Candlestick(x=time_list,
                                                  xaxis='x',
                                                  yaxis='y2',
                                                  name='价格',
                                                  open=cl[:, 0],
                                                  close=cl[:, 1],
                                                  high=cl[:, 2],
                                                  low=cl[:, 3],
                                                  increasing=dict(line=dict(color='#FF2131')),
                                                  decreasing=dict(line=dict(color='#00CCFF')))
                    """
                    py.offline.plot({
                        "data": [profit_scatter, reference_scatter, random_scatter, price_scatter, trade_bar,
                                 amount_scatter],
                        "layout": go.Layout(title=glo.stock_code + "回测结果",
                                            xaxis=dict(title='日期', type="category", showgrid=False, zeroline=False),
                                            yaxis=dict(title='收益率', showgrid=False, zeroline=False),
                                            yaxis2=dict(title='股价', overlaying='y', side='right',
                                                        titlefont={'color': 'orange'}, tickfont={'color': 'orange'},
                                                        showgrid=False,
                                                        zeroline=False),
                                            yaxis3=dict(title='交易量', overlaying='y', side='right',
                                                        titlefont={'color': '#000099'}, tickfont={'color': '#000099'},
                                                        showgrid=False, position=0.97, zeroline=False, anchor='free'),
                                            yaxis4=dict(title='持股量', overlaying='y', side='left',
                                                        titlefont={'color': '#00ccff'}, tickfont={'color': '#00ccff'},
                                                        showgrid=False, position=0.03, zeroline=False, anchor='free'),
                                            paper_bgcolor='#FFFFFF',
                                            plot_bgcolor='#FFFFFF',
                                            )
                    }, auto_open=False, filename=path)

            step_reward += reward[0][0]
            current_stock_state = next_stock_state
            current_agent_state = next_agent_state
            # 更新date
            date_manager.next_date()
            print(
                "此时刻全部卖出可净收入：" + str(glo.get_stock_total_value(glo.price) + glo.money - glo.ori_money - glo.ori_value))
            print("----------------------------")
            if t % 5000 == 0 and t != 0 and (train_model == "train" or train_model == "both"):
                save_weights()
        print("-------------------------------------------------------------------------------------------")
        print("total_reward:" + str(step_reward))
        if train_model == "train" or train_model == "both":
            main_actor_net.save_weights('sim_res_weights/' + str(episode + 1) + '_main_actor_weights.h5',
                                        overwrite=True)
            target_actor_net.save_weights('sim_res_weights/' + str(episode + 1) + '_target_actor_weights.h5',
                                          overwrite=True)
            main_critic_net.save_weights('sim_res_weights/' + str(episode + 1) + '_main_critic_weights.h5',
                                         overwrite=True)
            target_critic_net.save_weights('sim_res_weights/' + str(episode + 1) + '_target_critic_weights.h5',
                                           overwrite=True)
            save_weights()
        if (episode + 1) % 10 == 0 and (train_model == "train" or train_model == "both"):
            save_experience_pool()


def save_weights():
    main_actor_net.save_weights('main_actor_weights.h5', overwrite=True)
    target_actor_net.save_weights('target_actor_weights.h5', overwrite=True)
    main_critic_net.save_weights('main_critic_weights.h5', overwrite=True)
    target_critic_net.save_weights('target_critic_weights.h5', overwrite=True)
    print("权重存储完成")


def save_experience_pool():
    print("经验存储中......请勿退出!!!")
    with open("Data/experience_pool.json", "w", encoding='UTF-8') as f:
        json.dump(experience_pool, f, default=lambda obj: obj.__dict__)
    print("经验存储完成")


run_model(input("请输入运行模式：run\\train\\both\n"))
