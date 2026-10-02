import os
import secrets
import string
import time
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROWS = 6
COLS = 7
RED = 'red'
YELLOW = 'yellow'
EMPTY = None

ROOM_CODE_CHARS = string.ascii_uppercase + string.digits
ROOM_TTL_SECONDS = 30 * 60


# ── Game state ───────────────────────────────────────────────────────
def new_game_state():
    return {
        'board': [[EMPTY] * COLS for _ in range(ROWS)],
        'current_player': RED,
        'game_over': False,
        'winner': None,
        'win_cells': [],
        'is_draw': False,
        'players': {},        # color -> player_token (rooms only)
        'last_active': time.time(),
    }


# Single shared game, used by the existing vs-computer mode.
ai_state = new_game_state()

# Online 2-player games, keyed by short room code.
rooms = {}


# ── Game logic (operates on a state dict, shared by AI mode & rooms) ──
def get_target_row(state, col):
    board = state['board']
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] is EMPTY:
            return r
    return -1


def check_win_for(state, row, col, player):
    board = state['board']
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        line = [(row, col)]
        for s in range(1, 4):
            r, c = row + dr * s, col + dc * s
            if not (0 <= r < ROWS and 0 <= c < COLS) or board[r][c] != player:
                break
            line.append((r, c))
        for s in range(1, 4):
            r, c = row - dr * s, col - dc * s
            if not (0 <= r < ROWS and 0 <= c < COLS) or board[r][c] != player:
                break
            line.append((r, c))
        if len(line) >= 4:
            return line
    return None


def get_valid_cols(state):
    return [c for c in range(COLS) if get_target_row(state, c) != -1]


def score_window(window, player):
    """Score a window of 4 cells for the given player."""
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


def score_board(state):
    """Heuristic score of the board from YELLOW's perspective."""
    board = state['board']
    score = 0

    # Centre column preference
    centre = COLS // 2
    centre_array = [board[r][centre] for r in range(ROWS)]
    score += centre_array.count(YELLOW) * 6

    # Horizontal windows
    for r in range(ROWS):
        for c in range(COLS - 3):
            w = [board[r][c + i] for i in range(4)]
            score += score_window(w, YELLOW)

    # Vertical windows
    for c in range(COLS):
        for r in range(ROWS - 3):
            w = [board[r + i][c] for i in range(4)]
            score += score_window(w, YELLOW)

    # Diagonal ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            w = [board[r + i][c + i] for i in range(4)]
            score += score_window(w, YELLOW)

    # Diagonal ↙
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            w = [board[r + i][c - i] for i in range(4)]
            score += score_window(w, YELLOW)

    return score


def minimax(state, depth, alpha, beta, maximising):
    """Minimax with alpha-beta pruning. Returns (best_col, score)."""
    valid_cols = get_valid_cols(state)
    board = state['board']

    if not valid_cols:
        return None, 0  # draw

    if depth == 0:
        return None, score_board(state)

    # Move ordering: try centre columns first (improves pruning)
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
                return col, 100000 + depth   # win sooner = better
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
                return col, -(100000 + depth)  # human win = bad for computer
            _, score = minimax(state, depth - 1, alpha, beta, True)
            board[row][col] = EMPTY
            if score < best_score:
                best_score = score
                best_col = col
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_col, best_score


def computer_pick_col(state):
    col, _ = minimax(state, 5, float('-inf'), float('inf'), True)
    return col if col is not None else -1


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
        state['current_player'] = YELLOW if player == RED else RED
    return row


# ── API helpers ──────────────────────────────────────────────────────
def state_dict(state, computer_col=None):
    return {
        'board': state['board'],
        'currentPlayer': state['current_player'],
        'gameOver': state['game_over'],
        'winner': state['winner'],
        'winCells': state['win_cells'],
        'draw': state['is_draw'],
        'computerCol': computer_col,
    }


# ── Routes: vs-computer mode (unchanged behaviour) ─────────────────────
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

    place_piece(ai_state, col, RED)

    computer_col = None
    if not ai_state['game_over']:
        computer_col = computer_pick_col(ai_state)
        if computer_col != -1:
            place_piece(ai_state, computer_col, YELLOW)

    return jsonify(state_dict(ai_state, computer_col))


@app.route('/reset', methods=['POST'])
def reset():
    global ai_state
    ai_state = new_game_state()
    return jsonify(state_dict(ai_state))


# ── Routes: online 2-player rooms ──────────────────────────────────────
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


def room_response(room, color, computer_col=None):
    sd = state_dict(room, computer_col)
    sd['yourColor'] = color
    sd['opponentJoined'] = len(room['players']) == 2
    sd['isYourTurn'] = (not room['game_over']) and room['current_player'] == color
    return jsonify(sd)


@app.route('/api/rooms', methods=['POST'])
def create_room():
    cleanup_rooms()
    room_id = generate_room_code()
    token = secrets.token_hex(8)
    room = new_game_state()
    room['players'] = {RED: token}
    rooms[room_id] = room
    return jsonify({'roomId': room_id, 'playerToken': token, 'color': RED})


@app.route('/api/rooms/<room_id>/join', methods=['POST'])
def join_room(room_id):
    room = find_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404
    if len(room['players']) >= 2:
        return jsonify({'error': 'Room is full'}), 400
    token = secrets.token_hex(8)
    room['players'][YELLOW] = token
    return jsonify({'roomId': room_id.upper(), 'playerToken': token, 'color': YELLOW})


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
    if len(room['players']) < 2:
        return jsonify({'error': 'Waiting for opponent'}), 400
    if room['game_over'] or room['current_player'] != color:
        return jsonify({'error': 'Not your turn'}), 400

    col = data.get('col')
    if col is None or not (0 <= col < COLS):
        return jsonify({'error': 'Invalid column'}), 400
    if get_target_row(room, col) == -1:
        return jsonify({'error': 'Column full'}), 400

    place_piece(room, col, color)
    return room_response(room, color)


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
    fresh = new_game_state()
    fresh['players'] = players
    rooms[room_id.upper()] = fresh
    return room_response(fresh, color)


# ── Start server ─────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f'Connect 4 running at http://localhost:{port}')
    app.run(host='0.0.0.0', port=port)
