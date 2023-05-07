"""棋盘游戏控制"""
import numpy as np
import copy
from collections import deque  # 这个队列用来判断长将或长捉

from config import CONFIG

# 列表来表示棋盘，红方在上，黑方在下。使用时需要使用深拷贝
state_list_init = [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '红炮', '一一', '一一', '一一', '一一', '一一', '红炮', '一一'],
                   ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
                   ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']]
opposite_dict = {
    '黑': '红',
    '红': '黑'
}
state_guy_init = ['红车', '红马', '红马', '红车', '红炮', '红炮', '红兵', '红兵', '红兵', '红兵', '红兵', '黑兵', '黑兵', '黑兵', '黑兵', '黑兵',
                  '黑炮', '黑炮', '黑车', '黑马', '黑马', '黑车']
# deque来存储棋盘状态，长度为4
state_deque_init = deque(maxlen=4)
for _ in range(4):
    state_deque_init.append(copy.deepcopy(state_list_init))

# 构建一个字典：字符串到数组的映射，函数：数组到字符串的映射
string2array = dict(红车=np.array([1, 0, 0, 0, 0, 0, 0]), 红马=np.array([0, 1, 0, 0, 0, 0, 0]),
                    红象=np.array([0, 0, 1, 0, 0, 0, 0]), 红士=np.array([0, 0, 0, 1, 0, 0, 0]),
                    红帅=np.array([0, 0, 0, 0, 1, 0, 0]), 红炮=np.array([0, 0, 0, 0, 0, 1, 0]),
                    红兵=np.array([0, 0, 0, 0, 0, 0, 1]), 黑车=np.array([-1, 0, 0, 0, 0, 0, 0]),
                    黑马=np.array([0, -1, 0, 0, 0, 0, 0]), 黑象=np.array([0, 0, -1, 0, 0, 0, 0]),
                    黑士=np.array([0, 0, 0, -1, 0, 0, 0]), 黑帅=np.array([0, 0, 0, 0, -1, 0, 0]),
                    黑炮=np.array([0, 0, 0, 0, 0, -1, 0]), 黑兵=np.array([0, 0, 0, 0, 0, 0, -1]),
                    一一=np.array([0, 0, 0, 0, 0, 0, 0]))


def array2string(array):
    return list(filter(lambda string: (string2array[string] == array).all(), string2array))[0]


# 改变棋盘状态
def change_state(state_list, move):
    """move : 字符串'0010'"""
    copy_list = copy.deepcopy(state_list)
    y, x, toy, tox = int(move[0]), int(move[1]), int(move[2]), int(move[3])
    copy_list[toy][tox] = copy_list[y][x]
    copy_list[y][x] = '一一'
    return copy_list


# 打印盘面，可视化用到
def print_board(_state_array):
    # _state_array: [10, 9, 7], HWC
    board_line = []
    for i in range(10):
        for j in range(9):
            board_line.append(array2string(_state_array[i][j]))
        print(board_line)
        board_line.clear()


# 列表棋盘状态到数组棋盘状态
def state_list2state_array(state_list):
    _state_array = np.zeros([10, 9, 7])
    for i in range(10):
        for j in range(9):
            _state_array[i][j] = string2array[state_list[i][j]]
    return _state_array


# 拿到所有合法走子的集合，2086长度，也就是神经网络预测的走子概率向量的长度
# 第一个字典：move_id到move_action
# 第二个字典：move_action到move_id
# 例如：move_id:0 --> move_action:'0010'
def get_all_legal_moves():
    _move_id2move_action = {}
    _move_action2move_id = {}
    row = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # 士的全部走法
    advisor_labels = ['0314', '1403', '0514', '1405', '2314', '1423', '2514', '1425',
                      '9384', '8493', '9584', '8495', '7384', '8473', '7584', '8475']
    # 象的全部走法
    bishop_labels = ['2002', '0220', '2042', '4220', '0224', '2402', '4224', '2442',
                     '2406', '0624', '2446', '4624', '0628', '2806', '4628', '2846',
                     '7052', '5270', '7092', '9270', '5274', '7452', '9274', '7492',
                     '7456', '5674', '7496', '9674', '5678', '7856', '9678', '7896']
    idx = 0
    for l1 in range(10):
        for n1 in range(9):
            destinations = [(t, n1) for t in range(10)] + \
                           [(l1, t) for t in range(9)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]  # 马走日
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(10) and n2 in range(9):
                    action = column[l1] + row[n1] + column[l2] + row[n2]
                    _move_id2move_action[idx] = action
                    _move_action2move_id[action] = idx
                    idx += 1

    for action in advisor_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1

    for action in bishop_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1

    return _move_id2move_action, _move_action2move_id


# move_id2move_action, move_action2move_id = get_all_legal_moves()
with open(CONFIG['move_id_dict'], 'r') as f:
    move_action2move_id = eval(f.read())
with open(CONFIG['id_move_dict'], 'r') as f:
    move_id2move_action = eval(f.read())


# 走子翻转的函数，用来扩充我们的数据
def flip_map(string):
    new_str = ''
    for index in range(4):
        if index == 0 or index == 2:
            new_str += (str(string[index]))
        else:
            new_str += (str(8 - int(string[index])))
    return new_str


# 边界检查
def check_bounds(toY, toX):
    if toY in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] and toX in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        return True
    return False


# 不能走到自己的棋子位置
def check_obstruct(piece, current_player_color):
    # 当走到的位置存在棋子的时候，进行一次判断
    if piece != '一一':
        if current_player_color == '红':
            if '黑' in piece:
                return True
            else:
                return False
        elif current_player_color == '黑':
            if '红' in piece:
                return True
            else:
                return False
    else:
        return True


def che_sub(state, y, x, raw, col, color, moves):
    v = state[raw][col]
    if v != '一一':
        cur_color = v[0]
        if cur_color != color:
            moves.append(str(y) + str(x) + str(raw) + str(col))
            return moves, False
        return moves, False
    else:
        moves.append(str(y) + str(x) + str(raw) + str(col))
        return moves, True


# y行10 x列9
def che(y, x, state, color):
    moves = []
    for i in range(y - 1, -1, -1):
        moves, label = che_sub(state, y, x, i, x, color, moves)
        if not label:
            break
    for i in range(y + 1, 10):
        moves, label = che_sub(state, y, x, i, x, color, moves)
        if not label:
            break
    for i in range(x - 1, -1, -1):
        moves, label = che_sub(state, y, x, y, i, color, moves)
        if not label:
            break
    for i in range(x + 1, 9):
        moves, label = che_sub(state, y, x, y, i, color, moves)
        if not label:
            break
    return moves


def che_one(y, x, state, color):
    moves = []
    for i in range(y - 1, -1, -1):
        moves, label = che_sub(state, y, x, i, x, color, moves)
        if not label:
            break
    for i in range(y + 1, 10):
        moves, label = che_sub(state, y, x, i, x, color, moves)
        if not label:
            break
    return moves


def ma(y, x, state, color):
    def get_raw_plus(state, px, py):
        piece_close = state[px + 1][py]
        if piece_close != '一一':
            return False
        return True

    def get_raw_minus(state, px, py):
        piece_close = state[px - 1][py]
        if piece_close != '一一':
            return False
        return True

    def get_col_plus(state, px, py):
        piece_close = state[px][py + 1]
        if piece_close != '一一':
            return False
        return True

    def get_col_minus(state, px, py):
        piece_close = state[px][py - 1]
        if piece_close != '一一':
            return False
        return True

    def get_default(state, px, py):
        return True

    temp_dict = {
        2: get_raw_plus,
        -2: get_raw_minus,
        6: get_col_plus,
        -6: get_col_minus
    }
    px, py = y, x
    places = [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]
    moves = []
    for place in places:
        x = px + place[0]
        y = py + place[1]
        if 0 <= x <= 9 and 0 <= y <= 8:
            f1 = temp_dict.get(place[0], get_default)(state, px, py)
            f2 = temp_dict.get(place[1] * 3, get_default)(state, px, py)
            if f1 and f2:
                piece_other = state[x][y]
                if color != piece_other[0]:
                    moves.append(str(px) + str(py) + str(x) + str(y))
    return moves


xiang_dict = {(2, 0): [['2002', [0, 2], [1, 1]], ['2042', [4, 2], [3, 1]]],
              (0, 2): [['0220', [2, 0], [1, 1]], ['0224', [2, 4], [1, 3]]],
              (4, 2): [['4220', [2, 0], [3, 1]], ['4224', [2, 4], [3, 3]]],
              (2, 4): [['2402', [0, 2], [1, 3]], ['2442', [4, 2], [3, 3]], ['2406', [0, 6], [1, 5]],
                       ['2446', [4, 6], [3, 5]]],
              (0, 6): [['0624', [2, 4], [1, 5]], ['0628', [2, 8], [1, 7]]],
              (4, 6): [['4624', [2, 4], [3, 5]], ['4628', [2, 8], [3, 7]]],
              (2, 8): [['2806', [0, 6], [1, 7]], ['2846', [4, 6], [3, 7]]],
              (7, 0): [['7052', [5, 2], [6, 1]], ['7092', [9, 2], [8, 1]]],
              (5, 2): [['5270', [7, 0], [6, 1]], ['5274', [7, 4], [6, 3]]],
              (9, 2): [['9270', [7, 0], [8, 1]], ['9274', [7, 4], [8, 3]]],
              (7, 4): [['7452', [5, 2], [6, 3]], ['7492', [9, 2], [8, 3]], ['7456', [5, 6], [6, 5]],
                       ['7496', [9, 6], [8, 5]]],
              (5, 6): [['5674', [7, 4], [6, 5]], ['5678', [7, 8], [6, 7]]],
              (9, 6): [['9674', [7, 4], [8, 5]], ['9678', [7, 8], [8, 7]]],
              (7, 8): [['7856', [5, 6], [6, 7]], ['7896', [9, 6], [8, 7]]]}

shi_dict = {(0, 3): [['0314', [1, 4]]],
            (1, 4): [['1403', [0, 3]], ['1405', [0, 5]], ['1423', [2, 3]], ['1425', [2, 5]]],
            (0, 5): [['0514', [1, 4]]],
            (2, 3): [['2314', [1, 4]]],
            (2, 5): [['2514', [1, 4]]],
            (9, 3): [['9384', [8, 4]]],
            (8, 4): [['8493', [9, 3]], ['8495', [9, 5]], ['8473', [7, 3]], ['8475', [7, 5]]],
            (9, 5): [['9584', [8, 4]]],
            (7, 3): [['7384', [8, 4]]],
            (7, 5): [['7584', [8, 4]]]}

shuai_dict = {(0, 4): [['0414', [1, 4]], ['0403', [0, 3]], ['0405', [0, 5]]],
              (0, 3): [['0313', [1, 3]], ['0304', [0, 4]]],
              (0, 5): [['0515', [1, 5]], ['0504', [0, 4]]],
              (1, 4): [['1404', [0, 4]], ['1424', [2, 4]], ['1413', [1, 3]], ['1415', [1, 5]]],
              (1, 3): [['1303', [0, 3]], ['1323', [2, 3]], ['1314', [1, 4]]],
              (1, 5): [['1505', [0, 5]], ['1525', [2, 5]], ['1514', [1, 4]]],
              (2, 4): [['2414', [1, 4]], ['2423', [2, 3]], ['2425', [2, 5]]],
              (2, 3): [['2313', [1, 3]], ['2324', [2, 4]]],
              (2, 5): [['2515', [1, 5]], ['2524', [2, 4]]],
              (9, 4): [['9484', [8, 4]], ['9493', [9, 3]], ['9495', [9, 5]]],
              (9, 3): [['9383', [8, 3]], ['9394', [9, 4]]],
              (9, 5): [['9585', [8, 5]], ['9594', [9, 4]]],
              (8, 4): [['8474', [7, 4]], ['8494', [9, 4]], ['8483', [8, 3]], ['8485', [8, 5]]],
              (8, 3): [['8373', [7, 3]], ['8393', [9, 3]], ['8384', [8, 4]]],
              (8, 5): [['8575', [7, 5]], ['8595', [9, 5]], ['8584', [8, 4]]],
              (7, 4): [['7484', [8, 4]], ['7473', [7, 3]], ['7475', [7, 5]]],
              (7, 3): [['7383', [8, 3]], ['7374', [7, 4]]],
              (7, 5): [['7585', [8, 5]], ['7574', [7, 4]]]}


def xiang(y, x, state, color):
    # 一共十七个位置，固定的，直接用字典
    moves = []
    move_list = xiang_dict.get((y, x), [])
    for i in move_list:
        move, end_pos, mid_pos = i
        if state[mid_pos[0], mid_pos[1]] == '一一' and state[end_pos[0], end_pos[1]][0] != color:
            moves.append(move)
    return moves


def shi(y, x, state, color):
    moves = []
    move_list = shi_dict.get((y, x), [])
    for i in move_list:
        move, end_pos = i
        if state[end_pos[0], end_pos[1]][0] != color:
            moves.append(move)
    return moves


def shuai(y, x, state, color):
    moves = []
    move_list = shuai_dict.get((y, x), [])
    opponent = opposite_dict[color] + '帅'
    for i in move_list:
        move, end_pos = i
        py, px = end_pos
        if state[py][px][0] != color:
            # 搜索王的位置，判断是否王对王，中间存在一个棋子就跳出
            other = 'none'
            if color == '红':
                for j in range(py + 1, 10):
                    guy = state[j][px]
                    if guy != '一一':
                        other = guy
                        break
                if other != opponent:
                    moves.append(move)
            else:
                for j in range(py - 1, -1, -1):
                    guy = state[j][px]
                    if guy != '一一':
                        other = guy
                        break
                if other != opponent:
                    moves.append(move)
    return moves


def da_bin(y, x, state, color):
    moves = []
    if color == '红':
        if y <= 4:
            raw = y + 1
            col = x
            if raw <= 9 and state[raw][col][0] != color:
                moves.append(str(y) + str(x) + str(raw) + str(col))
        else:
            for i in [[0, -1], [0, 1], [1, 0]]:
                raw = y + i[0]
                col = x + i[1]
                if raw <= 9 and -1 < col < 9 and state[raw][col][0] != color:
                    moves.append(str(y) + str(x) + str(raw) + str(col))
    elif color == '黑':
        if y > 4:
            raw = y - 1
            col = x
            if raw >= 0 and state[raw][col][0] != color:
                moves.append(str(y) + str(x) + str(raw) + str(col))
        else:
            for i in [[0, -1], [0, 1], [-1, 0]]:
                raw = y + i[0]
                col = x + i[1]
                if raw > -1 and -1 < col < 9 and state[raw][col][0] != color:
                    moves.append(str(y) + str(x) + str(raw) + str(col))
    return moves


def da_bin_one(y, x, state, color):
    moves = []
    if color == '红':
        if y <= 4:
            raw = y + 1
            col = x
            if raw <= 9 and state[raw][col][0] != color:
                moves.append(str(y) + str(x) + str(raw) + str(col))
        else:
            for i in [[1, 0]]:
                raw = y + i[0]
                col = x + i[1]
                if raw <= 9 and state[raw][col][0] != color:
                    moves.append(str(y) + str(x) + str(raw) + str(col))
    elif color == '黑':
        if y > 4:
            raw = y - 1
            col = x
            if raw >= 0 and state[raw][col][0] != color:
                moves.append(str(y) + str(x) + str(raw) + str(col))
        else:
            for i in [[-1, 0]]:
                raw = y + i[0]
                col = x + i[1]
                if raw >= 0 and state[raw][col][0] != color:
                    moves.append(str(y) + str(x) + str(raw) + str(col))
    return moves


def pao(y, x, state, color):
    opposite = opposite_dict[color]
    moves = []
    for i in range(y - 1, -1, -1):
        if state[i][x] == '一一':
            moves.append(str(y) + str(x) + str(i) + str(x))
        else:
            for j in range(i - 1, -1, -1):
                v = state[j][x]
                if v[0] == opposite:
                    moves.append(str(y) + str(x) + str(j) + str(x))
                    break
                elif v[0] != '一':
                    break
            break
    for i in range(y + 1, 10):
        if state[i][x] == '一一':
            moves.append(str(y) + str(x) + str(i) + str(x))
        else:
            for j in range(i + 1, 10):
                v = state[j][x]
                if v[0] == opposite:
                    moves.append(str(y) + str(x) + str(j) + str(x))
                    break
                elif v[0] != '一':
                    break
            break
    for i in range(x - 1, -1, -1):
        if state[y][i] == '一一':
            moves.append(str(y) + str(x) + str(y) + str(i))
        else:
            for j in range(i - 1, -1, -1):
                v = state[y][j]
                if v[0] == opposite:
                    moves.append(str(y) + str(x) + str(y) + str(j))
                    break
                elif v[0] != '一':
                    break
            break
    for i in range(x + 1, 9):
        if state[y][i] == '一一':
            moves.append(str(y) + str(x) + str(y) + str(i))
        else:
            for j in range(i + 1, 9):
                v = state[y][j]
                if v[0] == opposite:
                    moves.append(str(y) + str(x) + str(y) + str(j))
                    break
                elif v[0] != '一':
                    break
            break
    return moves


def pao_one(y, x, state, color):
    opposite_dict = {
        '黑': '红',
        '红': '黑'
    }
    opposite = opposite_dict[color]
    moves = []
    for i in range(y - 1, -1, -1):
        if state[i][x] == '一一':
            moves.append(str(y) + str(x) + str(i) + str(x))
        else:
            for j in range(i - 1, -1, -1):
                v = state[j][x]
                if v[0] == opposite:
                    moves.append(str(y) + str(x) + str(j) + str(x))
                    break
                elif v[0] != '一':
                    break
            break
    for i in range(y + 1, 10):
        if state[i][x] == '一一':
            moves.append(str(y) + str(x) + str(i) + str(x))
        else:
            for j in range(i + 1, 10):
                v = state[j][x]
                if v[0] == opposite:
                    moves.append(str(y) + str(x) + str(j) + str(x))
                    break
                elif v[0] != '一':
                    break
            break
    return moves


def where_shuai(state, color):
    # 给默认值避免判断出错
    top_pos = [0, 1]
    below_pos = [0, 2]

    pos_list = [4, 3, 5]
    for i in [0, 1, 2]:
        for j in pos_list:
            if state[i][j][1] == '帅':
                top_pos = [i, j]
                break
    for i in [9, 8, 7]:
        for j in pos_list:
            if state[i][j][1] == '帅':
                below_pos = [i, j]
                break
    shuai_move = []
    face_to_face = False
    if top_pos[1] == below_pos[1]:
        face_to_face = True
        col = below_pos[1]
        for i in range(top_pos[0] + 1, below_pos[0]):
            guy = state[i][col]
            if guy != '一一':
                face_to_face = False
                break
    if face_to_face:
        if color == '红':
            shuai_move.append(str(top_pos[0]) + str(top_pos[1]) + str(below_pos[0]) + str(below_pos[1]))
        else:
            shuai_move.append(str(below_pos[0]) + str(below_pos[1]) + str(top_pos[0]) + str(top_pos[1]))
    return shuai_move, top_pos, below_pos


def empty(y, x, state, color):
    return []


func_dict = {
    '车': che,
    '马': ma,
    '象': xiang,
    '士': shi,
    '帅': shuai,
    '兵': da_bin,
    '炮': pao,
}


def get_legal_moves(state_deque, current_player_color):
    color = current_player_color
    state = state_deque[-1]
    moves = []
    state = copy.deepcopy(state)
    shuai_move, top_pos, below_pos = where_shuai(state, color)
    if shuai_move:
        moves = [move_action2move_id.get(shuai_move[0], '')]
        return moves
    state = np.array(state)
    for y in range(10):
        for x in range(9):
            if state[y][x][0] == color:
                move_per = func_dict.get(state[y][x][1:], empty)(y, x, state, color)
                for i in move_per:
                    if state[int(i[2])][int(i[3])][1] == '帅':
                        return [move_action2move_id.get(i, '')]
                    else:
                        moves.append(move_action2move_id.get(i, ''))
    return moves


# 棋盘逻辑控制
class Board(object):

    def __init__(self):
        self.kill_action = 0
        self.state_list = copy.deepcopy(state_list_init)
        self.game_start = False
        self.winner = None
        self.state_deque = copy.deepcopy(state_deque_init)
        self.have_pos = copy.deepcopy(state_guy_init)
        self.last_move = None
        self.start_player = 1

        self.id2color = {1: '红', 2: '黑'}
        self.color2id = {'红': 1, '黑': 2}
        self.backhand_player = 2

        self.current_player_color = self.id2color[self.start_player]  # 红
        self.current_player_id = self.color2id['红']
        self.action_count = 0

    # 初始化棋盘的方法
    def init_board(self):  # 传入先手玩家的id
        # 当前手玩家，也就是先手玩家
        self.current_player_color = self.id2color[self.start_player]  # 红
        self.current_player_id = self.color2id['红']
        # 初始化棋盘状态
        self.state_list = copy.deepcopy(state_list_init)
        self.state_deque = copy.deepcopy(state_deque_init)
        # 记录游戏中吃子的回合数
        self.kill_action = 0
        self.game_start = False
        self.action_count = 0  # 游戏动作计数器

    # 获的当前盘面的所有合法走子集合
    @property
    def availables(self):
        moves = get_legal_moves(self.state_deque, self.current_player_color)
        return moves

    def current_map(self):
        return self.state_deque[-1], self.current_player_color

    # 从当前玩家的视角返回棋盘状态，current_state_array: [9, 10, 9] 走一步 CHW
    def current_state(self):
        _current_state = np.zeros([9, 10, 9])
        # 使用9个平面来表示棋盘状态
        # 0-6个平面表示棋子位置，1代表红方棋子，-1代表黑方棋子, 队列最后一个盘面
        # 第7个平面表示对手player最近一步的落子位置，走子之前的位置为-1，走子之后的位置为1，其余全部是0
        # 第8个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0
        _current_state[:7] = state_list2state_array(self.state_deque[-1]).transpose([2, 0, 1])  # [7, 10, 9]

        if self.game_start:
            # 解构self.last_move
            move = move_id2move_action[self.last_move]
            start_position = int(move[0]), int(move[1])
            end_position = int(move[2]), int(move[3])
            _current_state[7][start_position[0]][start_position[1]] = -1
            _current_state[7][end_position[0]][end_position[1]] = 1
        # 指出当前是哪个玩家走子
        if self.action_count % 2 == 0:
            _current_state[8][:, :] = 1.0

        return _current_state

    # 根据move对棋盘状态做出改变
    def do_move(self, move):
        self.game_start = True  # 游戏开始
        self.action_count += 1  # 移动次数加1
        move_action = move_id2move_action[move]
        start_y, start_x = int(move_action[0]), int(move_action[1])
        end_y, end_x = int(move_action[2]), int(move_action[3])
        state_list = copy.deepcopy(self.state_deque[-1])
        # 判断是否吃子
        if state_list[end_y][end_x] != '一一':
            end_pos = state_list[end_y][end_x]
            # 如果吃掉对方的帅，则返回当前的current_player胜利
            if end_pos == '红帅':
                self.winner = self.color2id['黑']
            elif end_pos == '黑帅':
                self.winner = self.color2id['红']
            elif end_pos in self.have_pos:
                self.have_pos.remove(end_pos)
        else:
            self.kill_action += 1
        # 更改棋盘状态
        state_list[end_y][end_x] = state_list[start_y][start_x]
        state_list[start_y][start_x] = '一一'
        self.current_player_color = '黑' if self.current_player_color == '红' else '红'  # 改变当前玩家
        self.current_player_id = 1 if self.current_player_id == 2 else 2
        # 记录最后一次移动的位置
        self.last_move = move
        self.state_deque.append(np.array(state_list))

    # 是否产生赢家
    def has_a_winner(self):
        """一共有三种状态，红方胜，黑方胜，平局"""
        if self.winner is not None:
            return True, self.winner
        elif self.kill_action >= CONFIG['kill_action'] or not self.have_pos:  # 平局先手判负
            return False, -1
        return False, -1

    # 检查当前棋局是否结束
    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif self.kill_action >= CONFIG['kill_action']:  # 平局，没有赢家
            return True, -1
        return False, -1

    def get_current_player_color(self):
        return self.current_player_color

    def get_current_player_id(self):
        return self.current_player_id


# 在Board类基础上定义Game类，该类用于启动并控制一整局对局的完整流程，并收集对局过程中的数据，以及进行棋盘的展示
class Game(object):

    def __init__(self, board):
        self.board = board
        self.play_with_ucci = CONFIG['play_with_ucci']

    # 用于人机对战，人人对战等
    def start_play(self, player1, player2, process_id=None, winner_dict=None):
        self.board.init_board()  # 初始化棋盘
        states, mcts_probs, current_players, moves = [], [], [], []
        p1, p2 = 1, 2
        player1.set_player_ind(1)
        player2.set_player_ind(2)
        players = {p1: player1, p2: player2}
        while True:
            current_player = self.board.get_current_player_id()  # 红子对应的玩家id
            player_polict_func = players[current_player]  # 决定当前玩家的代理
            move, move_probs = player_polict_func.get_action(self.board)  # 当前玩家代理拿到动作

            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player_id)
            moves.append(move_id2move_action[move])

            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = -1.0
                    if winner_dict:
                        final_winner = winner_dict[winner]
                    else:
                        final_winner = players[winner]
                    print(f"Game end. Winner is {final_winner}, Process_id: {process_id}")
                else:
                    print(f"Game end. Tie, Process_id: {process_id}")
                log_info = f'ai vs ai: evaluate'
                return winner, zip(states, mcts_probs, winner_z), moves, log_info

    # 使用蒙特卡洛树搜索开始自我对弈，存储游戏状态（状态，蒙特卡洛落子概率，胜负手）三元组用于神经网络训练
    # 初始化棋盘, start_player=1
    def start_self_play(self, player, is_shown=False, temp=1e-3, ucci_pos=1):
        self.board.init_board()
        states, mcts_probs, current_players, moves = [], [], [], []
        # 开始自我对弈
        _count = 0
        while True:
            if self.play_with_ucci and _count % 2 == ucci_pos:
                play_with_ucci = True
            else:
                play_with_ucci = False
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1, play_with_ucci=play_with_ucci)
            _count += 1
            # 保存自我对弈的数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player_id)
            moves.append(move_id2move_action[move])

            # 打印观测
            # board_map, color_cur = self.board.current_map()
            # print(self.board.current_player_id, self.board.current_player_color, play_with_ucci)
            # print(move_id2move_action[move])
            # for i in board_map:
            #     print(i)
            # print('\n')

            # 执行一步落子
            self.board.do_move(move)
            end, winner = self.board.game_end()

            if end:
                # 从每一个状态state对应的玩家的视角保存胜负信息
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = -1.0
                # 重置蒙特卡洛根节点
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is:", winner)
                    else:
                        print('Game end. Tie')
                log_info = f'ai vs ai: self'
                if self.play_with_ucci:
                    if winner != ucci_pos + 1:
                        log_info = 'ai vs ucci: ai 赢了'
                    else:
                        log_info = 'ai vs ucci: ai 输了'
                return winner, zip(states, mcts_probs, winner_z), moves, log_info


if __name__ == '__main__':
    board = Board()
    board.init_board()
    print(move_action2move_id)
