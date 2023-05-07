from collections import deque
import os
import pickle
import time
from threading import Thread
import queue
import datetime
import uuid

import requests

from lib.mcts import MCTSPlayer
from lib.game import Board, Game
from lib.ucci import get_ucci_move_func
from lib.log import debug_log
from lib import memory_array
from config import CONFIG
from model import PolicyValueNet


class CollectPipeline:

    def __init__(self):
        # 对弈参数
        self.n_playout = CONFIG['play_out']
        self.c_puct = CONFIG['c_puct']
        self.buffer_size = CONFIG['buffer_size']
        self.play_out = CONFIG['play_out']
        self.model_version_path = CONFIG['model_version_path']
        self.model_version = CONFIG['model_version']
        self.current_model_path = CONFIG['current_model_path']
        self.use_ucci = CONFIG['use_ucci']
        self.play_with_ucci = CONFIG['play_with_ucci']
        self.one_piece_path = CONFIG['one_piece_path']
        self.save_buffer_path = CONFIG['train_data_path']
        self.train_flag = CONFIG['train_flag']

        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.temp = 1
        self.policy_value_net = PolicyValueNet(model_file=self.current_model_path)
        self.mcts_player = None

    def download_func(self, data):
        file_name = data['file_name']
        best_version = data['best_version']
        print('开始下载模型')
        data = requests.post(url=f"{CONFIG['ip_address']}/download_model/{file_name}", json=data)
        if data.status_code == 200:
            with open(self.current_model_path, 'wb') as f:
                f.write(data.content)
            print(f'已下载模型：{file_name}，模型版本：{best_version}')
        else:
            print('下载模型失败')
        with open(self.model_version_path, 'w') as f:
            f.write(best_version)
        self.model_version = best_version
        self.save_buffer_path = f"{CONFIG['one_piece_path']}/{self.model_version}"
        if not os.path.exists(self.save_buffer_path):
            os.makedirs(self.save_buffer_path)
        return best_version

    # 检查是否为最新模型
    def download_model(self, input_dict):
        return_dict = input_dict[0]
        self.model_version = open(self.model_version_path, 'r').read()
        file_name = self.current_model_path.split('/')[-1]
        data = {
            'file_name': file_name,
            'model_version': self.model_version
        }
        data = requests.post(url=f"{CONFIG['ip_address']}/search_model", json=data).json()
        model_dir = data.get('model_dir', self.model_version)
        if data['msg'] != 'true':
            if data['best_version']:
                downloading = return_dict.get('download_time', False)
                if downloading:
                    print('模型在下载中不用继续下载')
                else:
                    self.model_version = open(self.model_version_path, 'r').read()
                    if self.model_version != data['best_version']:
                        return_dict['download_time'] = True
                        best_version = self.download_func(data)
                        return_dict['model_version'] = best_version
                        self.model_version = best_version
                        self.load_model(input_dict)
                        return_dict['download_time'] = False
            else:
                print('暂无模型')
        return model_dir

    def check_train(self):
        if os.path.exists(self.train_flag):
            with open(self.train_flag, 'r') as f:
                res = f.read()
            if res == 'start':
                print('训练中，暂停')
                time.sleep(60 * 10)
                return self.check_train()
        return True

    # 从主体加载模型
    def load_model(self, input_dict):
        return_dict = input_dict[0]
        self.model_version = return_dict.get('model_version', open(self.model_version_path, 'r').read())
        self.save_buffer_path = f"{CONFIG['one_piece_path']}/{self.model_version}"

        if not os.path.exists(self.save_buffer_path):
            os.makedirs(self.save_buffer_path)

        if os.path.exists(self.current_model_path):
            model_path = self.current_model_path
        else:
            model_path = None
        self.policy_value_net = PolicyValueNet(model_file=model_path)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
        return_dict['policy_value_net'] = self.policy_value_net
        print('已加载最新模型')

    @staticmethod
    def get_equi_data(play_data):
        extend_data = []
        # shape is [9, 10, 9], 走子概率，赢家
        for state, mcts_prob, winner in play_data:
            extend_data.append(memory_array.zip_state_mcts_prob((state, mcts_prob, winner)))
        return extend_data

    @staticmethod
    def upload_buffer(real_path, file_name, model_version):
        files_list = []
        t = ("files", (file_name, open(real_path, 'rb').read()))
        files_list.append(t)
        upload_res = requests.post(f'{CONFIG["ip_address"]}/upload_buffer/{model_version}',
                                   files=files_list).json()
        return upload_res

    # 批量上传
    def upload_buffer_all(self):
        files_list = []
        for i in os.listdir(self.one_piece_path):
            real_path = os.path.join(self.one_piece_path, i)
            t = ("files", (i, open(real_path, 'rb').read()))
            files_list.append(t)
        upload_res = requests.post(f'{CONFIG["ip_address"]}/upload_buffer/{self.model_version}',
                                   files=files_list).json()
        print(upload_res)

    def save_buffer_upload(self, model_version, file_name, data_dict):
        self.save_buffer_path = f"{CONFIG['one_piece_path']}/{model_version}"
        if not os.path.exists(self.save_buffer_path):
            os.makedirs(self.save_buffer_path)
        real_path = f'{self.save_buffer_path}/{file_name}'
        with open(real_path, 'wb') as f:
            pickle.dump(data_dict, f)
        try:
            upload_res = self.upload_buffer(real_path, file_name, model_version)
        except Exception as e:
            upload_res = {'code': '上传失败', 'filenames': [file_name]}
            print(e)
        return upload_res

    def check_version(self, input_dict):
        model_version = open(self.model_version_path, 'r').read()
        current_version = self.model_version.split('_')[0]
        if model_version != current_version:
            self.load_model(input_dict)
            self.model_version = model_version

    # 收集自我对弈的数据
    def collect_selfplay_data(self, process_id, game, input_list):
        self.check_train()
        self.check_version(input_list)
        model_dir = self.download_model(input_list)
        return_dict = input_list[0]
        policy_value_net = return_dict.get('policy_value_net', None)
        if policy_value_net:
            print(f'已获取最新模型: {self.model_version}')
            self.policy_value_net = policy_value_net

        if self.use_ucci:
            polict_func = get_ucci_move_func
        else:
            polict_func = self.policy_value_net.policy_value_fn

        if self.play_with_ucci:
            is_selfplay = 0
        else:
            is_selfplay = 1

        mcts_player = MCTSPlayer(polict_func,
                                 c_puct=self.c_puct,
                                 n_playout=self.n_playout,
                                 is_selfplay=is_selfplay)
        winner, play_data, moves, log_info = game.start_self_play(mcts_player, temp=self.temp, is_shown=True,
                                                                  ucci_pos=process_id % 2)
        play_data = list(play_data)[:]
        play_data = self.get_equi_data(play_data)[:]
        now_time = str(datetime.datetime.today()).split('.')[0]
        file_uid = uuid.uuid1()
        data_dict = {'play_data': play_data, 'time': now_time,
                     'moves': moves, 'winner': winner, 'log_info': log_info,
                     'file_uid': file_uid, 'user_id': CONFIG['user_id'],
                     'hostname': CONFIG['hostname'], 'model_version': model_dir,
                     'model_path': CONFIG['current_model_path']
                     }
        file_name = f'{file_uid}.pkl'
        debug_log.debug(log_info + f' file_name:{file_name}, winner:{winner}, date: {now_time.split()[0]}')
        upload_res = self.save_buffer_upload(model_dir, file_name, data_dict)
        return len(play_data), upload_res

    def run(self, process_id, return_dict):
        s_time = time.time()
        board = Board()
        game = Game(board)
        episode_len, upload_res = self.collect_selfplay_data(process_id, game, return_dict)
        upload_info = f"{upload_res['code']}, Filenames: {upload_res['filenames'][0]}"
        e_time = time.time()
        print(
            f'Process_id: {process_id}, Batch number: {process_id}, Episode_len: {episode_len}, Time: {round(e_time - s_time, 2)}, Upload_status: {upload_info}')


def main(i, collecting_pipeline, input_list):
    collecting_pipeline.run(i, input_list)


class Store(Thread):
    def __init__(self, store, queue):
        Thread.__init__(self)
        self.queue = queue
        self.store = store

    def run(self):
        try:
            main(self.store[0], self.store[1], self.store[2])
        except Exception as e:
            print(e)
        finally:
            self.queue.get()
            self.queue.task_done()


def start_thread():
    collecting_pipeline = CollectPipeline()
    q = queue.Queue(CONFIG['thread_nums'])
    input_list = [{}]
    for i in range(10000):
        q.put(i)
        t = Store((i, collecting_pipeline, input_list), q)
        t.start()
    q.join()
    print('over')


def start_one():
    collecting_pipeline = CollectPipeline()
    input_list = [{}]
    for i in range(10000):
        main(i, collecting_pipeline, input_list)
    print('over')


if __name__ == '__main__':
    start_thread()
