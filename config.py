import socket
import os
import torch
import uuid

hostname = socket.gethostname()
absolute_path = os.path.abspath(__file__).replace('\\', '/')
project_path = '/'.join(absolute_path.split('/')[0:-1])

if torch.cuda.is_available():
    device = 'cuda'
    use_gpu = True
else:
    device = 'cpu'
    use_gpu = False

# 配置ucci可执行文件路径
ucci_list = ['XQWizard/ELEEYE.EXE', 'xqcyclone/cyclone.exe']
eleeye_path_list = [f'{project_path}/models/ucci/{i}' for i in ucci_list]

CONFIG = {
    'kill_action': 180,  # 和棋回合数
    'dirichlet': 0.3,  # 随机程度 国际象棋，0.3 日本将棋，0.15 围棋，0.03
    'play_out': 800,  # 每次移动的模拟次数
    'c_puct': 1.5,  # u的权重
    'buffer_size': 1300000,  # 经验池大小
    'current_model_path': f'{project_path}/models/current_model/best_policy.pkl',
    'best_model_path': f'{project_path}/models/best_model/best_policy.pkl',
    'batch_size': 512,  # 每次更新的train_step数量
    'kl_targ': 0.02,  # kl散度控制
    'epochs': 5,  # 每次更新的train_step数量
    'n_games': 10,  # 每次评估场次
    'one_piece_path': f'{project_path}/data/play_data',  # 保存棋谱位置
    'ip_address': 'https://algorithm.randforest.cn',
    'device': device,
    'use_gpu': use_gpu,
    'hostname': hostname,
    'user_info_dir': f'{project_path}/config/',
    'elo_path': f'{project_path}/config/elo.txt',
    'model_version_path': f'{project_path}/config/model_version.txt',
    'train_flag': f'{project_path}/config/train.txt',
    'move_id_dict': f'{project_path}/config/move_id_dict.txt',
    'id_move_dict': f'{project_path}/config/id_move_dict.txt',
    'log_path': f'{project_path}/logs/',
    'thread_nums': 20,
    'eleeye_path': f'{project_path}/models/ucci/XQWizard/ELEEYE.EXE',
    'eleeye_path_list': eleeye_path_list,
    'think_time': 3,  # ucci思考时间/秒
    'use_ucci': False,  # 只用ucci生成棋谱
    'play_with_ucci': True,  # 只用ai和ucci对战生成棋谱
    'only_train': True
}


def dir_init():
    # 初始化模型保存地址
    model_current_dir = f'{project_path}/models/current_model/'
    if not os.path.exists(model_current_dir):
        os.makedirs(model_current_dir)
    model_best_dir = f'{project_path}/models/best_model/'
    if not os.path.exists(model_best_dir):
        os.makedirs(model_best_dir)

    # 初始化日志文件
    if not os.path.exists(CONFIG['log_path']):
        os.makedirs(CONFIG['log_path'])
    debug_log_path = os.path.join(CONFIG['log_path'], 'debug.log')
    error_log_path = os.path.join(CONFIG['log_path'], 'error.log')
    operation_log_path = os.path.join(CONFIG['log_path'], 'operation.log')
    CONFIG['debug_log_path'] = debug_log_path
    CONFIG['error_log_path'] = error_log_path
    CONFIG['operation_log_path'] = operation_log_path

    # 初始化ELO文件
    if not os.path.exists(CONFIG['elo_path']):
        with open(CONFIG['elo_path'], 'w') as f:
            f.write('1300\n1300')

    # 初始化棋谱根目录
    if not os.path.exists(CONFIG['one_piece_path']):
        os.makedirs(CONFIG['one_piece_path'])

    # 初始化棋谱版本
    if not os.path.exists(CONFIG['model_version_path']):
        with open(CONFIG['model_version_path'], 'w') as f:
            f.write('v1')
    with open(CONFIG['model_version_path'], 'r') as f:
        model_version = f.read()
    CONFIG['model_version'] = model_version

    # 初始化用户名
    user_info_dir = CONFIG['user_info_dir']
    user_info_file = os.path.join(user_info_dir, 'username.txt')
    if not os.path.exists(user_info_file):
        user_id = str(uuid.uuid1())
        if not os.path.exists(user_info_dir):
            os.makedirs(user_info_dir)
        with open(user_info_file, 'w') as f:
            f.write(user_id)
    else:
        with open(user_info_file, 'r') as f:
            user_id = f.read()
    CONFIG['user_id'] = user_id

    # 初始化当前模型棋谱地址
    train_data_path = os.path.join(CONFIG['one_piece_path'], CONFIG['model_version'])
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    CONFIG['train_data_path'] = train_data_path

    print(f'cuda: {use_gpu}')


dir_init()
