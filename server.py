import os
import secrets
import string
import time
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROWS = 6          # default row count when none is requested
COLS = 7           # fixed — only row count is configurable
RED = 'red'
YELLOW = 'yellow'
BLUE = 'blue'
GREEN = 'green'
EMPTY = None

TWO_PLAYER = [RED, YELLOW]
THREE_PLAYER = [RED, YELLOW, BLUE]
FOUR_PLAYER = [RED, YELLOW, BLUE, GREEN]
PLAYER_ORDERS = {2: TWO_PLAYER, 3: THREE_PLAYER, 4: FOUR_PLAYER}
ALLOWED_ROWS = {6, 8, 10}

ROOM_CODE_CHARS = string.ascii_uppercase + string.digits
ROOM_TTL_SECONDS = 30 * 60


# ── Game state ───────────────────────────────────────────────────────
def new_game_state(turn_order=None, rows=None):
    order = list(turn_order or TWO_PLAYER)
    row_count = rows or ROWS
    return {
        'board': [[EMPTY] * COLS for _ in range(row_count)],
        'turn_order': order,
        'current_player': order[0],
        'game_over': False,
        'winner': None,
        'win_cells': [],
        'is_draw': False,
        'players': {},        # color -> player_token (rooms only)
        'last_active': time.time(),
    }


# Single shared game, used by the vs-computer mode.
ai_state = new_game_state()

# Online games (2-4 human players), keyed by short room code.
rooms = {}


# ── Game logic (operates on a state dict, shared by AI mode & rooms) ──
# Row count is read from the state's own board (len(state['board'])) so
# each game can have a different height; only column count is fixed.
def get_target_row(state, col):
    board = state['board']
    for r in range(len(board) - 1, -1, -1):
        if board[r][col] is EMPTY:
            return r
    return -1


def check_win_for(state, row, col, player):
    board = state['board']
    rows = len(board)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        line = [(row, col)]
        for s in range(1, 4):
            r, c = row + dr * s, col + dc * s
            if not (0 <= r < rows and 0 <= c < COLS) or board[r][c] != player:
                break
            line.append((r, c))
        for s in range(1, 4):
            r, c = row - dr * s, col - dc * s
            if not (0 <= r < rows and 0 <= c < COLS) or board[r][c] != player:
                break
            line.append((r, c))
        if len(line) >= 4:
            return line
    return None


def get_valid_cols(state):
    return [c for c in range(COLS) if get_target_row(state, c) != -1]


def iter_windows(board):
    """Yield every 4-in-a-row window on the board (horizontal/vertical/both diagonals)."""
    rows = len(board)
    for r in range(rows):
        for c in range(COLS - 3):
            yield [board[r][c + i] for i in range(4)]
    for c in range(COLS):
        for r in range(rows - 3):
            yield [board[r + i][c] for i in range(4)]
    for r in range(rows - 3):
        for c in range(COLS - 3):
            yield [board[r + i][c + i] for i in range(4)]
    for r in range(rows - 3):
        for c in range(3, COLS):
            yield [board[r + i][c - i] for i in range(4)]


# ── 2-player AI: minimax with alpha-beta pruning ───────────────────────
def score_window_2p(window, player):
    opp = RED if player == YELLOW else YELLOW
    score = 0
    if window.count(player) == 4:
        score += 100000
    elif window.count(player) == 3 and window.count(EMPTY) == 1:
        score += 60
    elif window.count(player) == 2 and window.count(EMPTY) == 2:
        score += 10
    if window.count(opp) == 3 and window.count(EMPTY) == 1:
        score -= 80
    return score


def score_board_2p(state):
    """Heuristic score of the board from YELLOW's perspective."""
    board = state['board']
    score = 0
    centre = COLS // 2
    centre_array = [board[r][centre] for r in range(len(board))]
    score += centre_array.count(YELLOW) * 6
    for w in iter_windows(board):
        score += score_window_2p(w, YELLOW)
    return score


def minimax(state, depth, alpha, beta, maximising):
    """Minimax with alpha-beta pruning. Returns (best_col, score)."""
    valid_cols = get_valid_cols(state)
    board = state['board']

    if not valid_cols:
        return None, 0  # draw

    if depth == 0:
        return None, score_board_2p(state)

    centre = COLS // 2
    valid_cols.sort(key=lambda c: abs(centre - c))

    if maximising:  # Computer (YELLOW) wants highest score
        best_score = float('-inf')
        best_col = valid_cols[0]
        for col in valid_cols:
            row = get_target_row(state, col)
            board[row][col] = YELLOW
            if check_win_for(state, row, col, YELLOW):
                board[row][col] = EMPTY
                return col, 100000 + depth
            _, score = minimax(state, depth - 1, alpha, beta, False)
            board[row][col] = EMPTY
            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_col, best_score

    else:  # Human (RED) wants lowest score
        best_score = float('inf')
        best_col = valid_cols[0]
        for col in valid_cols:
            row = get_target_row(state, col)
            board[row][col] = RED
            if check_win_for(state, row, col, RED):
                board[row][col] = EMPTY
                return col, -(100000 + depth)
            _, score = minimax(state, depth - 1, alpha, beta, True)
            board[row][col] = EMPTY
            if score < best_score:
                best_score = score
                best_col = col
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_col, best_score


def computer_pick_col_2p(state):
    col, _ = minimax(state, 5, float('-inf'), float('inf'), True)
    return col if col is not None else -1


# ── 3+ player AI: maxⁿ search (generalizes minimax to n players) ──────
# Each node scores every player independently (no zero-sum assumption);
# the player to move picks the child that maximises their own score.
# No pruning is implemented, so the depth is kept shallow to stay fast.
MAXN_DEPTH = 4


def score_window_np(window, player):
    score = 0
    player_count = window.count(player)
    empty_count = window.count(EMPTY)
    if player_count == 4:
        score += 100000
    elif player_count == 3 and empty_count == 1:
        score += 60
    elif player_count == 2 and empty_count == 2:
        score += 10
    for opp in set(w for w in window if w and w != player):
        if window.count(opp) == 3 and empty_count == 1:
            score -= 80
    return score


def score_board_for(state, player):
    board = state['board']
    score = 0
    centre = COLS // 2
    centre_array = [board[r][centre] for r in range(len(board))]
    score += centre_array.count(player) * 6
    for w in iter_windows(board):
        score += score_window_np(w, player)
    return score


def maxn(state, depth, turn_order, turn_idx):
    """Returns (best_col, {color: score, ...}) for the player to move."""
    valid_cols = get_valid_cols(state)
    board = state['board']
    current = turn_order[turn_idx]

    if not valid_cols:
        return None, {p: 0 for p in turn_order}

    if depth == 0:
        return None, {p: score_board_for(state, p) for p in turn_order}

    centre = COLS // 2
    valid_cols.sort(key=lambda c: abs(centre - c))
    next_idx = (turn_idx + 1) % len(turn_order)

    best_col = valid_cols[0]
    best_scores = None
    for col in valid_cols:
        row = get_target_row(state, col)
        board[row][col] = current
        if check_win_for(state, row, col, current):
            scores = {p: (100000 + depth if p == current else -50000) for p in turn_order}
        else:
            _, scores = maxn(state, depth - 1, turn_order, next_idx)
        board[row][col] = EMPTY
        if best_scores is None or scores[current] > best_scores[current]:
            best_scores = scores
            best_col = col
    return best_col, best_scores


def computer_pick_col_np(state):
    turn_order = state['turn_order']
    turn_idx = turn_order.index(state['current_player'])
    col, _ = maxn(state, MAXN_DEPTH, turn_order, turn_idx)
    return col if col is not None else -1


def computer_pick_col(state):
    if len(state['turn_order']) >= 3:
        return computer_pick_col_np(state)
    return computer_pick_col_2p(state)


def place_piece(state, col, player):
    row = get_target_row(state, col)
    if row == -1:
        return -1
    state['board'][row][col] = player
    wc = check_win_for(state, row, col, player)
    if wc:
        state['game_over'] = True
        state['winner'] = player
        state['win_cells'] = wc
    elif all(state['board'][0][c] is not EMPTY for c in range(COLS)):
        state['game_over'] = True
        state['is_draw'] = True
    else:
        order = state['turn_order']
        idx = order.index(player)
        state['current_player'] = order[(idx + 1) % len(order)]
    return row


# ── API helpers ──────────────────────────────────────────────────────
def state_dict(state, moves=None):
    return {
        'board': state['board'],
        'turnOrder': state['turn_order'],
        'currentPlayer': state['current_player'],
        'gameOver': state['game_over'],
        'winner': state['winner'],
        'winCells': state['win_cells'],
        'draw': state['is_draw'],
        'moves': moves or [],
    }


def parse_turn_order(data):
    return PLAYER_ORDERS.get(data.get('players'), TWO_PLAYER)


def parse_rows(data):
    rows = data.get('rows')
    return rows if rows in ALLOWED_ROWS else ROWS


# ── Routes: vs-computer mode ────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/state')
def get_state():
    return jsonify(state_dict(ai_state))


@app.route('/move', methods=['POST'])
def make_move():
    if ai_state['game_over'] or ai_state['current_player'] != RED:
        return jsonify({'error': 'Not your turn'}), 400

    col = request.json.get('col')
    if col is None or not (0 <= col < COLS):
        return jsonify({'error': 'Invalid column'}), 400
    if get_target_row(ai_state, col) == -1:
        return jsonify({'error': 'Column full'}), 400

    row = place_piece(ai_state, col, RED)
    moves = [{'row': row, 'col': col, 'color': RED}]

    while not ai_state['game_over'] and ai_state['current_player'] != RED:
        cp = ai_state['current_player']
        ai_col = computer_pick_col(ai_state)
        if ai_col == -1:
            break
        ai_row = place_piece(ai_state, ai_col, cp)
        moves.append({'row': ai_row, 'col': ai_col, 'color': cp})

    return jsonify(state_dict(ai_state, moves))


@app.route('/reset', methods=['POST'])
def reset():
    global ai_state
    data = request.get_json(silent=True) or {}
    ai_state = new_game_state(parse_turn_order(data), parse_rows(data))
    return jsonify(state_dict(ai_state))


# ── Routes: online rooms (2-4 human players) ────────────────────────
def generate_room_code():
    while True:
        code = ''.join(secrets.choice(ROOM_CODE_CHARS) for _ in range(4))
        if code not in rooms:
            return code


def cleanup_rooms():
    cutoff = time.time() - ROOM_TTL_SECONDS
    stale = [rid for rid, r in rooms.items() if r['last_active'] < cutoff]
    for rid in stale:
        del rooms[rid]


def find_room(room_id):
    cleanup_rooms()
    room = rooms.get(room_id.upper())
    if room:
        room['last_active'] = time.time()
    return room


def get_player_color(room, token):
    for color, t in room['players'].items():
        if token and t == token:
            return color
    return None


def room_response(room, color, moves=None):
    sd = state_dict(room, moves)
    sd['yourColor'] = color
    sd['seatsTaken'] = len(room['players'])
    sd['seatsTotal'] = len(room['turn_order'])
    sd['allJoined'] = len(room['players']) == len(room['turn_order'])
    sd['isYourTurn'] = (not room['game_over']) and room['current_player'] == color
    return jsonify(sd)


@app.route('/api/rooms', methods=['POST'])
def create_room():
    cleanup_rooms()
    data = request.get_json(silent=True) or {}
    room_id = generate_room_code()
    token = secrets.token_hex(8)
    room = new_game_state(parse_turn_order(data), parse_rows(data))
    room['players'] = {room['turn_order'][0]: token}
    rooms[room_id] = room
    return jsonify({'roomId': room_id, 'playerToken': token, 'color': room['turn_order'][0]})


@app.route('/api/rooms/<room_id>/join', methods=['POST'])
def join_room(room_id):
    room = find_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404
    open_colors = [c for c in room['turn_order'] if c not in room['players']]
    if not open_colors:
        return jsonify({'error': 'Room is full'}), 400
    color = open_colors[0]
    token = secrets.token_hex(8)
    room['players'][color] = token
    return jsonify({'roomId': room_id.upper(), 'playerToken': token, 'color': color})


@app.route('/api/rooms/<room_id>/state')
def room_state(room_id):
    room = find_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404
    color = get_player_color(room, request.args.get('token'))
    if not color:
        return jsonify({'error': 'Invalid token'}), 403
    return room_response(room, color)


@app.route('/api/rooms/<room_id>/move', methods=['POST'])
def room_move(room_id):
    room = find_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.get_json(silent=True) or {}
    color = get_player_color(room, data.get('token'))
    if not color:
        return jsonify({'error': 'Invalid token'}), 403
    if len(room['players']) < len(room['turn_order']):
        return jsonify({'error': 'Waiting for players'}), 400
    if room['game_over'] or room['current_player'] != color:
        return jsonify({'error': 'Not your turn'}), 400

    col = data.get('col')
    if col is None or not (0 <= col < COLS):
        return jsonify({'error': 'Invalid column'}), 400
    if get_target_row(room, col) == -1:
        return jsonify({'error': 'Column full'}), 400

    row = place_piece(room, col, color)
    return room_response(room, color, [{'row': row, 'col': col, 'color': color}])


@app.route('/api/rooms/<room_id>/reset', methods=['POST'])
def room_reset(room_id):
    room = find_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.get_json(silent=True) or {}
    color = get_player_color(room, data.get('token'))
    if not color:
        return jsonify({'error': 'Invalid token'}), 403

    players = room['players']
    fresh = new_game_state(room['turn_order'], len(room['board']))
    fresh['players'] = players
    rooms[room_id.upper()] = fresh
    return room_response(fresh, color)


# ── Start server ─────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f'Connect 4 running at http://localhost:{port}')
    app.run(host='0.0.0.0', port=port)
