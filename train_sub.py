import random
from collections import deque
import numpy as np
import pickle
import time
import os
from collections import Counter
import datetime
import torch
import uuid
import traceback
from threading import Thread
import queue

import requests

from lib import memory_array
from config import CONFIG
from lib.game import Game, Board
from lib.mcts import MCTSPlayer
from model import PolicyValueNet
from lib.elo import elo_cal

CONFIG['use_ucci'] = False


# 定义整个训练流程
class TrainPipeline:

    def __init__(self):
        self.n_playout = CONFIG['play_out']
        self.c_puct = CONFIG['c_puct']
        self.batch_size = CONFIG['batch_size']
        self.epochs = CONFIG['epochs']
        self.kl_targ = CONFIG['kl_targ']
        self.buffer_size = CONFIG['buffer_size']
        self.one_piece_path = CONFIG['one_piece_path']
        self.model_version_path = CONFIG['model_version_path']
        self.model_version = CONFIG['model_version']
        self.train_data_path = CONFIG['train_data_path']
        self.current_model_path = CONFIG['current_model_path']
        self.elo_path = CONFIG['elo_path']
        self.train_flag = CONFIG['train_flag']
        self.n_games = CONFIG['n_games']

        # 训练参数
        self.board = Board()
        self.game = Game(self.board)
        self.learn_rate = 0.0001
        self.lr_multiplier = 1
        self.temp = 1.0
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.check_freq = 100
        self.total_train_times = 5001

        mcts_player, self.policy_value_net = self.load_model()
        print(f'本次训练评估频率：{self.check_freq}, 预计训练总次数：{self.total_train_times}')

    # 从主体加载模型
    def load_model(self):
        if os.path.exists(self.current_model_path):
            model_path = self.current_model_path
        else:
            model_path = None
        policy_value_net = PolicyValueNet(model_file=model_path)
        mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                 c_puct=self.c_puct,
                                 n_playout=self.n_playout,
                                 is_selfplay=1)
        return mcts_player, policy_value_net

    def read_elo(self):
        with open(self.elo_path) as f:
            res = f.read()
        current_elo, old_elo = res.split('\n')
        return float(current_elo), float(old_elo)

    def save_buffer_upload(self, model_version, play_data, moves, winner, log_info):
        play_data = list(play_data)[:]
        play_data = self.get_equi_data(play_data)
        now_time = str(datetime.datetime.today()).split('.')[0]
        file_uid = uuid.uuid1()
        data_dict = {'play_data': play_data, 'time': now_time,
                     'moves': moves, 'winner': winner, 'log_info': log_info,
                     'file_uid': file_uid, 'user_id': CONFIG['user_id'],
                     'hostname': CONFIG['hostname'], 'model_version': self.model_version,
                     'model_path': CONFIG['current_model_path']
                     }
        file_name = f'{file_uid}.pkl'
        self.one_piece_path = f"{CONFIG['one_piece_path']}/{model_version}"
        if not os.path.exists(self.one_piece_path):
            os.makedirs(self.one_piece_path)
        real_path = f'{self.one_piece_path}/{file_name}'
        with open(real_path, 'wb') as f:
            pickle.dump(data_dict, f)

    @staticmethod
    def get_equi_data(play_data):
        extend_data = []
        # shap: [9, 10, 9], 走子概率，赢家
        for state, mcts_prob, winner in play_data:
            extend_data.append(memory_array.zip_state_mcts_prob((state, mcts_prob, winner)))
        return extend_data

    def play_one(self, input_params):
        game, mcts_player_up, mcts_player_down, model_version, input_list, winner_dict = input_params
        winner, play_data, moves, log_info = game.start_play(mcts_player_up, mcts_player_down, winner_dict=winner_dict)
        input_list.append(winner)
        self.save_buffer_upload(model_version, play_data, moves, winner, log_info)

    def create_player(self, func_up, func_down):
        player_up = MCTSPlayer(func_up, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=0)
        player_down = MCTSPlayer(func_down, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=0)
        return player_up, player_down

    def evaluate_func(self, new_version):
        self.data_buffer = deque(maxlen=self.buffer_size)
        current_elo, old_elo = self.read_elo()
        policy_value_current = self.policy_value_net
        policy_value_best = PolicyValueNet(model_file=CONFIG['best_model_path'])
        func_current = policy_value_current.policy_value_fn
        func_best = policy_value_best.policy_value_fn
        res_list = []
        rating_num = self.n_games
        q = queue.Queue(10)
        input_list_up = []
        for i in range(rating_num):
            game = Game(board=Board())
            winner_dict = {
                1: 'current_model',
                2: 'best_model'
            }
            player_current, player_best = self.create_player(func_current, func_best)
            input_prams = (game, player_current, player_best, new_version + '_eval', input_list_up, winner_dict)
            q.put(i)
            t = Store(input_prams, q,
                      self.play_one)
            t.start()
        input_list_down = []
        for i in range(rating_num):
            winner_dict = {
                1: 'best_model',
                2: 'current_model'
            }
            game = Game(board=Board())
            player_current, player_best = self.create_player(func_best, func_current)
            input_prams = (game, player_current, player_best, new_version + '_eval', input_list_down, winner_dict)
            q.put(i)
            t = Store(input_prams, q,
                      self.play_one)
            t.start()
        q.join()
        for i in input_list_down:
            if i == 1:
                input_list_up.append(2)
            elif i == 2:
                input_list_up.append(1)
        for winner in input_list_up:
            if winner == 1:
                current_elo, old_elo = elo_cal(current_elo, old_elo, 1, 0, 32)
            elif winner == 2:
                current_elo, old_elo = elo_cal(current_elo, old_elo, 0, 1, 32)
            res_list.append(winner)
        win_cnt = dict(Counter(res_list))
        win_ratio = 1.0 * (win_cnt.get(1, 0) + 0.5 * win_cnt.get(-1, 0)) / len(res_list)
        print(win_ratio)
        if win_ratio >= 0.5:
            print('模型结果提升')
            policy_value_current.save_model(CONFIG['best_model_path'])
            print(f'新的最优模型已保存: {str(datetime.datetime.now())}')
            with open(self.elo_path, 'w') as f:
                f.write(f'{current_elo}\n{current_elo}')
        else:
            print('模型结果没提升')
            with open(self.elo_path, 'w') as f:
                f.write(f'{current_elo}\n{old_elo}')
        # 清空缓存
        policy_value_best.policy_value_fn = None
        self.train_end_flag()
        return win_ratio

    def policy_evaluate(self, no_evaluate=False):
        with open(self.model_version_path, 'r') as f:
            model_version = f.read()
        new_version = 'v' + str(int(model_version[1:]) + 1)
        if no_evaluate:
            print('再次开始训练')
            return False
        # 开始评估
        win_ratio = self.evaluate_func(new_version)
        # 更新模型版本
        with open(self.model_version_path, 'w') as f:
            f.write(new_version)
            self.model_version = new_version
        return win_ratio

    def policy_updata(self, train_times):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        mini_batch = [memory_array.recovery_state_mcts_prob(data) for data in mini_batch]
        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype('float32')

        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype('float32')

        # 旧的策略，旧的价值函数
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        kl = None
        new_v = None
        loss = None
        entropy = None
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate
            )
            # 新的策略，新的价值函数
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1))
            if kl > self.kl_targ * 4:
                break
        # [
        #     (0, 0.01),
        #     (150000, 0.003),
        #     (400000, 0.0001),
        # ]
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        print(train_times, ("kl:{:.5f},"
                            "lr_multiplier:{:.3f},"
                            "loss:{},"
                            "entropy:{},"
                            "explained_var_old:{:.9f},"
                            "explained_var_new:{:.9f}"
                            ).format(kl,
                                     self.lr_multiplier,
                                     loss,
                                     entropy,
                                     explained_var_old,
                                     explained_var_new))
        return loss, entropy

    # 检查是否为最新模型
    def download_model(self):
        file_name = self.current_model_path.split('/')[-1]
        data = {
            'file_name': file_name,
            'model_version': self.model_version
        }
        data = requests.post(url=f"{CONFIG['ip_address']}/search_model", json=data).json()
        if data['msg'] != 'true':
            if data['best_version']:
                file_name = data['file_name']
                best_version = data['best_version']
                data = requests.post(url=f"{CONFIG['ip_address']}/download_model/{file_name}", json=data)
                if data.status_code == 200:
                    with open(file_name, 'wb') as f:
                        f.write(data.content)
                    print(f'已下载模型：{file_name}，模型版本：{self.model_version}')
                else:
                    print('下载模型失败')
                with open(self.model_version_path, 'w') as f:
                    f.write(best_version)
                self.model_version = best_version
                self.one_piece_path = f"{CONFIG['one_piece_path']}/{self.model_version}"
                print(f'已下载最新模型：{file_name}，模型版本：{best_version}')
            else:
                print('暂无模型')

    def read_file(self, path):
        for j in os.listdir(path):
            buffer_path = os.path.join(path, j)
            with open(buffer_path, 'rb') as f:
                try:
                    data_file = pickle.load(f)
                    self.data_buffer.extend(data_file['play_data'])
                except Exception as e:
                    print(e, j)

    def read_buffer(self):
        with open(self.model_version_path, 'r') as f:
            self.model_version = f.read()

        print(f'开始读取模型版本棋谱：{self.model_version}')
        train_data_path = os.path.join(self.one_piece_path, self.model_version)
        self.read_file(train_data_path)
        print(len(self.data_buffer))

        temp_data_path = os.path.join(self.one_piece_path, self.model_version + '_temp')
        if os.path.exists(temp_data_path):
            print('从临时棋谱文件夹读取数据')
            self.read_file(temp_data_path)

        last_version = 'v' + str(int(self.model_version[1:]) - 1)
        eval_data_path = os.path.join(self.one_piece_path, last_version + '_eval')
        if os.path.exists(eval_data_path):
            print('从评估棋谱文件夹读取数据')
            self.read_file(eval_data_path)

        print(f'已读取数据：{len(self.data_buffer)}')

    def train_start_flag(self):
        with open(self.train_flag, 'w') as f:
            f.write('start')

    def train_end_flag(self):
        with open(self.train_flag, 'w') as f:
            f.write('end')

    def run(self, no_evaluate=False):
        """开始训练"""
        try:
            # 读取实际的模型版本
            with open(self.model_version_path, 'r') as f:
                model_version = f.read()
            temp_version = model_version + '_temp'
            self.read_buffer()
            if len(self.data_buffer) < self.batch_size * 50:
                print('数据不足')
                self.train_end_flag()
                return False
            self.train_start_flag()
            for i in range(self.total_train_times):
                self.policy_updata(i)
                if i % 500 == 0:
                    self.policy_value_net.save_model(CONFIG['current_model_path'])
            self.policy_evaluate(no_evaluate)
            if no_evaluate:
                self.policy_value_net = None
                torch.cuda.empty_cache()
        except KeyboardInterrupt:
            print('\n\rquit')

    def collect_buffer(self):
        self.download_model()
        self.read_buffer()

    def evaluate(self):
        self.policy_evaluate()


class Store(Thread):
    def __init__(self, store, queue, func):
        Thread.__init__(self)
        self.queue = queue
        self.store = store
        self.func = func

    def run(self):
        try:
            self.func(self.store)
        except Exception as e:
            print(e)
            print(traceback.print_exc())
        finally:
            self.queue.get()
            self.queue.task_done()


def train_main(only_collect=False, only_evaluate=False, no_evaluate=False):
    train_tp = TrainPipeline()
    if only_evaluate:
        with open(CONFIG['model_version_path'], 'r') as f:
            model_version = f.read()
        print(f'开始评估模型版本：{model_version}')
        train_tp.evaluate_func(model_version)
    elif only_collect:
        train_tp.collect_buffer()
    else:
        train_tp.run(no_evaluate)


if __name__ == '__main__':
    only_evaluate = False
    while True:
        try:
            train_main(only_evaluate=only_evaluate)
            only_evaluate = False
            time.sleep(3600 * 10)
        except Exception as e:
            print(e)
            print(traceback.print_exc())
