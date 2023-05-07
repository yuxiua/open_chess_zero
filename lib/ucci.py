import subprocess
import copy
import random

from config import CONFIG

with open(CONFIG['move_id_dict'], 'r') as f:
    move_action2move_id = eval(f.read())

eleeye_path_list = CONFIG['eleeye_path_list']
replace_dict_old = {
    'n': 'k',
    'N': 'K',
    'b': 'e',
    'B': 'E',
    'a': 'm',
    'A': 'M',
    'k': 's',
    'K': 'S',
    'r': 'r',
    'R': 'R',
    'p': 'p',
    'P': 'P',
    'c': 'c',
    'C': 'C',
}

replace_dict = {'红车': 'r', '红马': 'k', '红象': 'e', '红士': 'm', '红帅': 's', '一一': '', '红炮': 'c', '红兵': 'p', '黑兵': 'P',
                '黑炮': 'C', '黑车': 'R', '黑马': 'K', '黑象': 'E', '黑士': 'M', '黑帅': 'S'}

state_to_board_dict = {
    'k': 'n',
    'K': 'N',
    'e': 'b',
    'E': 'B',
    'm': 'a',
    'M': 'A',
    's': 'k',
    'S': 'K',
    'r': 'r',
    'R': 'R',
    'p': 'p',
    'P': 'P',
    'c': 'c',
    'C': 'C',
}

BOARD_WIDTH = 9
BOARD_HEIGHT = 10

board = [['r', 'n', 'b', 'a', 'k', 'a', 'b', 'n', 'r'], ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
         ['.', 'c', '.', '.', '.', '.', '.', 'c', '.'], ['p', '.', 'p', '.', 'p', '.', 'p', '.', 'p'],
         ['.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
         ['P', '.', 'P', '.', 'P', '.', 'P', '.', 'P'], ['.', 'C', '.', '.', '.', '.', '.', 'C', '.'],
         ['.', '.', '.', '.', '.', '.', '.', '.', '.'], ['R', 'N', 'B', 'A', 'K', 'A', 'B', 'N', 'R']]

board1 = [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
          ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
          ['一一', '红炮', '一一', '一一', '一一', '一一', '一一', '红炮', '一一'],
          ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
          ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
          ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
          ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
          ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
          ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
          ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']]


def state_to_fen(state, turns):
    state = "".join([state_to_board_dict[s] if s.isalpha() else s for s in state])
    if turns % 2 == 0:
        fen = state + f' w - - 0 {turns}'
    else:
        fen = state + f' b - - 0 {turns}'
    return fen


def fen_to_state(fen):
    foo = fen.split(' ')
    position = foo[0]
    state = "".join([replace_dict[s] if s.isalpha() else s for s in position])
    return state


def flip_fen(fen):
    foo = fen.split(' ')
    rows = foo[0].split('/')

    def swapcase(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a

    def swapall(aa):
        return "".join([swapcase(a) for a in aa])

    return "/".join([swapall(reversed(row)) for row in reversed(rows)]) \
           + " " + ('w' if foo[1] == 'b' else 'b') \
           + " " + foo[2] \
           + " " + foo[3] + " " + foo[4] + " " + foo[5]


def fliped_state(state):
    rows = state.split('/')

    def swapcase(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a

    def swapall(aa):
        return "".join([swapcase(a) for a in aa])

    return "/".join([swapall(reversed(row)) for row in reversed(rows)])


def swapcase(a, s2b=False):
    if a.isalpha():
        if s2b:
            a = state_to_board_dict[a]
        else:
            a = replace_dict[a]
        return a.lower() if a.isupper() else a.upper()
    return a


def state_to_board(state):
    board = [['.' for col in range(BOARD_WIDTH)] for row in range(BOARD_HEIGHT)]
    x = 0
    y = 9
    for k in range(0, len(state)):
        ch = state[k]
        if ch == ' ':
            break
        if ch == '/':
            x = 0
            y -= 1
        elif ch >= '1' and ch <= '9':
            for i in range(int(ch)):
                board[y][x] = '.'
                x = x + 1
        else:
            board[y][x] = swapcase(ch, s2b=True)
            x = x + 1
    return board


def board_to_state(board):
    c = 0
    fen = ''
    for i in range(BOARD_HEIGHT - 1, -1, -1):
        c = 0
        for j in range(BOARD_WIDTH):
            if board[i][j] == '一一':
                c = c + 1
            else:
                if c > 0:
                    fen = fen + str(c)
                fen = fen + swapcase(board[i][j])
                c = 0
        if c > 0:
            fen = fen + str(c)
        if i > 0:
            fen = fen + '/'

    # res = state_to_board(fen)
    # for i in res:
    #     print(i)
    return fen


def board_to_state_old(board):
    c = 0
    fen = ''
    for i in range(BOARD_HEIGHT - 1, -1, -1):
        c = 0
        for j in range(BOARD_WIDTH):
            if board[i][j] == '.':
                c = c + 1
            else:
                if c > 0:
                    fen = fen + str(c)
                fen = fen + swapcase(board[i][j])
                c = 0
        if c > 0:
            fen = fen + str(c)
        if i > 0:
            fen = fen + '/'
    return fen


def parse_ucci_move(move):
    x0, x1 = ord(move[0]) - ord('a'), ord(move[2]) - ord('a')
    move = move[1] + str(x0) + move[3] + str(x1)
    return move


def parse_ucci_move_old(move):
    x0, x1 = ord(move[0]) - ord('a'), ord(move[2]) - ord('a')
    move = str(x0) + move[1] + str(x1) + move[3]
    return move


def get_ucci_move(fen, time=3):
    if CONFIG['play_with_ucci']:
        eleeye_path = CONFIG['eleeye_path']
    else:
        eleeye_path = random.choice(eleeye_path_list)
    p = subprocess.Popen(eleeye_path,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         universal_newlines=True)
    setfen = f'position fen {fen}\n'
    setrandom = f'setoption randomness none\n'
    cmd = 'ucci\n' + setrandom + setfen + f'go time {time * 1000}\n'
    try:
        out, err = p.communicate(cmd, timeout=time + 0.5)
    except subprocess.TimeoutExpired:
        p.kill()
        try:
            out, err = p.communicate()
        except Exception as e:
            print(f"{e}, cmd = {cmd}")
            return get_ucci_move(fen, time + 1)
    lines = out.split('\n')
    if lines[-2] == 'nobestmove':
        return None
    move = lines[-2].split(' ')[1]
    if move == 'depth':
        move = lines[-1].split(' ')[6]

    move = parse_ucci_move(move)

    return move, lines


def get_ucci_move_func(board, color=None):
    think_time = CONFIG['think_time']
    board_map, color_cur = board.current_map()
    if color:
        color_cur = color
    if color_cur == '黑':
        turns = 1
    else:
        turns = 2
        think_time += 1
    state = board_to_state(copy.deepcopy(board_map))
    fen = state_to_fen(state, turns)
    move, lines = get_ucci_move(fen, time=think_time)
    return move_action2move_id.get(move, None), lines


def get_ucci_move_test(board, turns):
    state = board_to_state(board)
    fen = state_to_fen(state, turns)
    move, lines = get_ucci_move(fen, time=CONFIG['think_time'])
    return move, lines


if __name__ == '__main__':
    INIT_STATE = 'rkemsmekr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RKEMSMEKR'
    board = [['红车', '一一', '红象', '红士', '红帅', '红士', '红象', '红车', '一一'],
       ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
       ['红马', '一一', '红炮', '一一', '红炮', '一一', '红马', '一一', '一一'],
       ['红兵', '一一', '红兵', '一一', '红兵', '一一', '一一', '一一', '红兵'],
       ['一一', '一一', '一一', '一一', '一一', '一一', '红兵', '一一', '一一'],
       ['黑兵', '黑马', '黑兵', '一一', '一一', '一一', '一一', '一一', '一一'],
       ['一一', '一一', '一一', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
       ['一一', '黑炮', '一一', '一一', '一一', '一一', '黑马', '黑炮', '一一'],
       ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
       ['黑车', '一一', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑车', '一一']]
    state = INIT_STATE
    move, lines = get_ucci_move_test(board, 2)
    print(move)
    t = move_action2move_id[move]
    print(lines)
