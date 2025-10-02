# tetris_population_training.py
# Population-based Tetris training with NES-style scoring + Pygame visualization.
# Saves genomes to files in the 'genomes' folder and updates them as they evolve.
# Save and run: python tetris_population_training.py
# Requires: pip install pygame

import pygame
import threading
import time
import random
import math
import json
import os
from dataclasses import dataclass, field
from typing import List
from collections import defaultdict
from datetime import datetime

# ---------- CONFIG ----------
POPULATION_SIZE = 100
ROWS = 20
COLS = 10

WINDOW_W = 1400
WINDOW_H = 900
LEADERBOARD_WIDTH = 360
THUMB_MARGIN = 4

EVOLUTION_INTERVAL_SEC = 5.0
ELITE_COUNT = max(1, POPULATION_SIZE // 6)
MUTATION_RATE = 0.15
STATS_PRINT_INTERVAL = 5.0

GLOBAL_SLOWDOWN = 1.0   # multiply delays (use <1 to speed up)
LOCK_DELAY = 0.12       # simulated lock delay in seconds
MAX_STEPS_PER_GAME = 2500

GENOME_DIR = "genomes"  # where genomes are saved
BEST_GENOME_FILE = os.path.join(GENOME_DIR, "best_genome.json")
# ---------- END CONFIG ----------

# NES-style scoring table
SCORE_TABLE = {1: 40, 2: 100, 3: 300, 4: 1200}

# Colors / visuals
COLOR_BG = (22,22,22)
COLOR_GRID = (50,50,50)
COLOR_CELL_BG = (30,30,30)
COLORS = {
    'I': (0, 255, 255),
    'O': (255, 200, 0),
    'T': (160, 0, 255),
    'S': (0, 255, 0),
    'Z': (255, 0, 0),
    'J': (0, 0, 255),
    'L': (255, 140, 0),
    0: (15,15,15)
}
TEXT_COLOR = (230,230,230)

# Tetromino shapes (rotation sets)
PIECES = {
    'I': [
        [(0,0),(0,1),(0,2),(0,3)],
        [(0,2),(1,2),(2,2),(3,2)],
        [(1,0),(1,1),(1,2),(1,3)],
        [(0,1),(1,1),(2,1),(3,1)]
    ],
    'O': [
        [(0,0),(0,1),(1,0),(1,1)],
    ] * 4,
    'T': [
        [(0,1),(1,0),(1,1),(1,2)],
        [(0,1),(1,1),(1,2),(2,1)],
        [(1,0),(1,1),(1,2),(2,1)],
        [(0,1),(1,0),(1,1),(2,1)]
    ],
    'S': [
        [(0,1),(0,2),(1,0),(1,1)],
        [(0,1),(1,1),(1,2),(2,2)],
        [(1,1),(1,2),(2,0),(2,1)],
        [(0,0),(1,0),(1,1),(2,1)]
    ],
    'Z': [
        [(0,0),(0,1),(1,1),(1,2)],
        [(0,2),(1,1),(1,2),(2,1)],
        [(1,0),(1,1),(2,1),(2,2)],
        [(0,1),(1,0),(1,1),(2,0)]
    ],
    'J': [
        [(0,0),(1,0),(1,1),(1,2)],
        [(0,1),(0,2),(1,1),(2,1)],
        [(1,0),(1,1),(1,2),(2,2)],
        [(0,1),(1,1),(2,1),(2,0)]
    ],
    'L': [
        [(0,2),(1,0),(1,1),(1,2)],
        [(0,1),(1,1),(2,1),(2,2)],
        [(1,0),(1,1),(1,2),(2,0)],
        [(0,0),(0,1),(1,1),(2,1)]
    ]
}
PIECE_KEYS = list(PIECES.keys())

# ---------- Utilities ----------
def clamp(v,a,b): return max(a,min(b,v))
def copy_board(board): return [row.copy() for row in board]

def drop_height_on_board(board, shape, left_col):
    rows = len(board)
    for top in range(-6, rows):
        if not can_place_on_board(board, shape, top+1, left_col):
            return top
    return rows-1

def can_place_on_board(board, shape, top_row, left_col):
    rows = len(board); cols = len(board[0])
    for dr,dc in shape:
        r = top_row + dr; c = left_col + dc
        if c < 0 or c >= cols: return False
        if r >= rows: return False
        if r >= 0 and board[r][c] != 0: return False
    return True

def apply_shape_and_clear(board, shape, top_row, left_col, piece_id=9):
    newb = [row.copy() for row in board]
    rows = len(board); cols = len(board[0])
    for dr,dc in shape:
        r = top_row + dr; c = left_col + dc
        if 0 <= r < rows and 0 <= c < cols:
            newb[r][c] = piece_id
    cleared = 0
    out = []
    for r in range(rows):
        if all(newb[r][c] != 0 for c in range(cols)):
            cleared += 1
        else:
            out.append(newb[r])
    for _ in range(cleared):
        out.insert(0,[0]*cols)
    return newb if cleared==0 else out, cleared

def aggregate_height(board):
    rows = len(board); cols = len(board[0])
    heights = []
    for c in range(cols):
        h = 0
        for r in range(rows):
            if board[r][c] != 0:
                h = rows - r; break
        heights.append(h)
    return sum(heights), heights

def count_holes(board):
    rows = len(board); cols = len(board[0])
    holes = 0
    for c in range(cols):
        found = False
        for r in range(rows):
            if board[r][c] != 0:
                found = True
            elif found:
                holes += 1
    return holes

def bumpiness_from_heights(heights):
    return sum(abs(heights[i]-heights[i+1]) for i in range(len(heights)-1))

# ---------- TetrisModel ----------
class TetrisModel:
    def __init__(self, rows=ROWS, cols=COLS):
        self.rows = rows; self.cols = cols
        self.grid = [[0]*cols for _ in range(rows)]
        self.lines_cleared = 0
        self.level = 0
        self.game_over = False

    def reset(self):
        self.grid = [[0]*self.cols for _ in range(self.rows)]
        self.lines_cleared = 0
        self.level = 0
        self.game_over = False

    def spawn_piece(self, piece_type):
        origin_r = -2
        origin_c = (self.cols // 2) - 2
        return {'type': piece_type, 'rot': 0, 'r': origin_r, 'c': origin_c}

    def can_place(self, piece):
        shape = PIECES[piece['type']][piece['rot']]
        for dr,dc in shape:
            rr = piece['r'] + dr; cc = piece['c'] + dc
            if cc < 0 or cc >= self.cols: return False
            if rr >= self.rows: return False
            if rr >= 0 and self.grid[rr][cc] != 0: return False
        return True

    def lock_piece(self, piece):
        shape = PIECES[piece['type']][piece['rot']]
        for dr,dc in shape:
            rr = piece['r'] + dr; cc = piece['c'] + dc
            if 0 <= rr < self.rows and 0 <= cc < self.cols:
                self.grid[rr][cc] = piece['type']
            elif rr < 0:
                self.game_over = True
        cleared = 0
        new_grid = []
        for r in range(self.rows):
            if all(self.grid[r][c] != 0 for c in range(self.cols)):
                cleared += 1
            else:
                new_grid.append(self.grid[r])
        for _ in range(cleared):
            new_grid.insert(0, [0]*self.cols)
        self.grid = new_grid
        if cleared > 0:
            self.lines_cleared += cleared
            self.level = self.lines_cleared // 10
        return cleared

# ---------- Genome / Agent ----------
@dataclass
class Genome:
    w_height: float = field(default_factory=lambda: random.uniform(-0.8, 0.0))
    w_lines: float = field(default_factory=lambda: random.uniform(0.6, 1.2))
    w_holes: float = field(default_factory=lambda: random.uniform(-1.5, -0.1))
    w_bump: float = field(default_factory=lambda: random.uniform(-0.8, 0.0))
    speed: int = field(default_factory=lambda: random.randint(2,9))  # 1..10

    def mutate(self):
        if random.random() < MUTATION_RATE: self.w_height += random.gauss(0,0.12)
        if random.random() < MUTATION_RATE: self.w_lines += random.gauss(0,0.12)
        if random.random() < MUTATION_RATE: self.w_holes += random.gauss(0,0.12)
        if random.random() < MUTATION_RATE: self.w_bump += random.gauss(0,0.12)
        if random.random() < MUTATION_RATE: self.speed = clamp(self.speed + int(round(random.gauss(0,1.5))), 1, 10)

    def copy(self): return Genome(self.w_height, self.w_lines, self.w_holes, self.w_bump, self.speed)

    def to_dict(self):
        return {
            "w_height": float(self.w_height),
            "w_lines": float(self.w_lines),
            "w_holes": float(self.w_holes),
            "w_bump": float(self.w_bump),
            "speed": int(self.speed)
        }

    @staticmethod
    def from_dict(d):
        return Genome(
            w_height=float(d.get("w_height", d.get("h", 0.0))),
            w_lines=float(d.get("w_lines", d.get("l", 0.0))),
            w_holes=float(d.get("w_holes", d.get("ho", 0.0))),
            w_bump=float(d.get("w_bump", d.get("b", 0.0))),
            speed=int(d.get("speed", d.get("spd", 6)))
        )

@dataclass
class AgentState:
    id: int
    genome: Genome
    score: int = 0
    games_played: int = 0
    lines_cleared_total: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    board_snapshot: List[List] = field(default_factory=list)
    current_piece: dict = field(default_factory=dict)
    next_piece: str = ''
    game_over: bool = False

# ---------- Genome file management ----------
def ensure_genome_dir():
    try:
        os.makedirs(GENOME_DIR, exist_ok=True)
    except Exception as e:
        print("Warning: could not create genome dir:", e)

def genome_filename(agent_id:int):
    return os.path.join(GENOME_DIR, f"genome_{agent_id}.json")

def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    try:
        os.replace(tmp, path)
    except Exception:
        # fallback
        os.remove(tmp)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

def save_genome_file(agent: AgentState):
    """Write agent.genome to genomes/genome_{id}.json safely."""
    fname = genome_filename(agent.id)
    with agent.lock:
        try:
            payload = {
                "id": agent.id,
                "genome": agent.genome.to_dict(),
                "meta": {
                    "score": int(agent.score),
                    "games_played": int(agent.games_played),
                    "lines_cleared_total": int(agent.lines_cleared_total),
                    "updated": datetime.utcnow().isoformat() + "Z"
                }
            }
            atomic_write_json(fname, payload)
        except Exception as e:
            print(f"Error saving genome file {fname}: {e}")

def load_genome_file(agent_id:int):
    fname = genome_filename(agent_id)
    if not os.path.exists(fname):
        return None
    try:
        with open(fname, "r", encoding="utf-8") as f:
            payload = json.load(f)
        gdict = payload.get("genome", payload)
        return Genome.from_dict(gdict)
    except Exception as e:
        print(f"Warning: failed to load genome file {fname}: {e}")
        return None

def save_best_genome(genome: Genome, stats: dict=None):
    payload = {"genome": genome.to_dict(), "meta": {"saved": datetime.utcnow().isoformat() + "Z"}}
    if stats:
        payload["meta"].update(stats)
    atomic_write_json(BEST_GENOME_FILE, payload)

def save_all_genomes(agents: List[AgentState]):
    for a in agents:
        save_genome_file(a)

# ---------- Agent worker ----------
def agent_worker(agent: AgentState, stop_event: threading.Event):
    rng = random.Random(agent.id + int(time.time()))
    while not stop_event.is_set():
        model = TetrisModel(ROWS, COLS)
        def draw_piece(): return random.choice(PIECE_KEYS)
        current = model.spawn_piece(draw_piece())
        next_piece = draw_piece()
        cumulative_score = 0
        steps = 0
        lock_on_ground = False
        lock_start = None

        while not model.game_over and not stop_event.is_set() and steps < MAX_STEPS_PER_GAME:
            steps += 1
            best_val = -1e9; best_plan = None
            piece_type = current['type']
            for rot_idx in range(len(PIECES[piece_type])):
                shape = PIECES[piece_type][rot_idx]
                for left in range(-4, COLS+1):
                    top = drop_height_on_board(model.grid, shape, left)
                    if not can_place_on_board(model.grid, shape, top, left):
                        continue
                    newb, cleared = apply_shape_and_clear(model.grid, shape, top, left, piece_id=9)
                    agg, heights = aggregate_height(newb)
                    holes = count_holes(newb)
                    bump = bumpiness_from_heights(heights)
                    with agent.lock:
                        g = agent.genome
                        heuristic = g.w_height * agg + g.w_lines * cleared + g.w_holes * holes + g.w_bump * bump
                    nes_score = (SCORE_TABLE.get(cleared,0) * (model.level + 1))
                    val = heuristic + nes_score * 0.02
                    val += -abs((left+2) - (COLS//2)) * 0.02
                    if val > best_val:
                        best_val = val
                        best_plan = (rot_idx, left, top, cleared, newb)
            if best_plan is None:
                model.game_over = True
                break
            rot_idx, target_col, target_top, cleared, after_board = best_plan
            rotations_needed = (rot_idx - current['rot']) % len(PIECES[current['type']])
            for _ in range(rotations_needed):
                new_rot = (current['rot'] + 1) % len(PIECES[current['type']])
                test_piece = {'type': current['type'], 'rot': new_rot, 'r': current['r'], 'c': current['c']}
                if model.can_place(test_piece):
                    current['rot'] = new_rot
                with agent.lock:
                    agent.board_snapshot = copy_board(model.grid)
                    agent.current_piece = dict(current)
                    agent.next_piece = next_piece
                    agent.game_over = model.game_over
                tdel = max(0.01, 0.04 * (1.0 - (agent.genome.speed - 1)/9.0)) * GLOBAL_SLOWDOWN
                t0 = time.time()
                while time.time() - t0 < tdel and not stop_event.is_set():
                    time.sleep(0.005)
            move_attempts = 0
            while move_attempts < 30 and not stop_event.is_set():
                move_attempts += 1
                shape_now = PIECES[current['type']][current['rot']]
                cur_drop = drop_height_on_board(model.grid, shape_now, current['c'])
                if current['c'] == target_col or cur_drop == target_top:
                    break
                if current['c'] < target_col:
                    cand = {'type': current['type'], 'rot': current['rot'], 'r': current['r'], 'c': current['c']+1}
                    if model.can_place(cand):
                        current['c'] += 1
                elif current['c'] > target_col:
                    cand = {'type': current['type'], 'rot': current['rot'], 'r': current['r'], 'c': current['c']-1}
                    if model.can_place(cand):
                        current['c'] -= 1
                with agent.lock:
                    agent.board_snapshot = copy_board(model.grid)
                    agent.current_piece = dict(current)
                    agent.next_piece = next_piece
                    agent.game_over = model.game_over
                tdel = max(0.01, 0.04 * (1.0 - (agent.genome.speed - 1)/9.0)) * GLOBAL_SLOWDOWN
                t0 = time.time()
                while time.time() - t0 < tdel and not stop_event.is_set():
                    time.sleep(0.005)
            while not stop_event.is_set():
                cand = {'type': current['type'], 'rot': current['rot'], 'r': current['r']+1, 'c': current['c']}
                if model.can_place(cand):
                    current['r'] += 1
                    with agent.lock:
                        agent.board_snapshot = copy_board(model.grid)
                        agent.current_piece = dict(current)
                        agent.next_piece = next_piece
                        agent.game_over = model.game_over
                    tdel = max(0.01, 0.04 * (1.0 - (agent.genome.speed - 1)/9.0)) * GLOBAL_SLOWDOWN
                    t0 = time.time()
                    while time.time() - t0 < tdel and not stop_event.is_set():
                        time.sleep(0.005)
                    continue
                else:
                    if not lock_on_ground:
                        lock_on_ground = True
                        lock_start = time.time()
                    else:
                        if time.time() - lock_start >= LOCK_DELAY:
                            cleared_lines = model.lock_piece(current)
                            nes_add = SCORE_TABLE.get(cleared_lines, 0) * (model.level + 1)
                            cumulative_score += nes_add
                            with agent.lock:
                                agent.lines_cleared_total += cleared_lines
                            current = model.spawn_piece(next_piece)
                            next_piece = draw_piece = random.choice(PIECE_KEYS)
                            lock_on_ground = False
                            lock_start = None
                            break
                    time.sleep(0.005)
            cumulative_score += 0
            with agent.lock:
                agent.board_snapshot = copy_board(model.grid)
                agent.current_piece = dict(current) if not model.game_over else {}
                agent.next_piece = next_piece if not model.game_over else ''
                agent.game_over = model.game_over
            if steps > MAX_STEPS_PER_GAME:
                break

        with agent.lock:
            agent.games_played += 1
            agent.score += int(cumulative_score)
            agent.board_snapshot = copy_board(model.grid)
            agent.current_piece = {}
            agent.next_piece = ''
            agent.game_over = model.game_over
        time.sleep(0.12)

# ---------- Evolution manager ----------
def evolution_manager(agents: List[AgentState], stop_event: threading.Event):
    threads = []
    for a in agents:
        t = threading.Thread(target=agent_worker, args=(a, stop_event), daemon=True)
        t.start(); threads.append(t)
    last_evolve = time.time()
    last_stats = time.time()
    try:
        while not stop_event.is_set():
            now = time.time()
            if now - last_evolve >= EVOLUTION_INTERVAL_SEC:
                snapshot = sorted(agents, key=lambda ag: ag.score, reverse=True)
                # copy elites into worst slots
                for i in range(ELITE_COUNT):
                    src = snapshot[i]
                    dst = snapshot[-1 - i]
                    newg = src.genome.copy()
                    newg.mutate()
                    with dst.lock:
                        dst.genome = newg
                        dst.score = 0; dst.games_played = 0; dst.lines_cleared_total = 0
                    # save dst genome file after replacement
                    save_genome_file(dst)
                    print(f"Evolution: copied genome from ID {src.id} -> ID {dst.id} (saved genome_{dst.id}.json)")
                last_evolve = now
            if now - last_stats >= STATS_PRINT_INTERVAL:
                snapshot = sorted(agents, key=lambda ag: ag.score, reverse=True)
                print("\n=== Leaderboard (Top {}) ===".format(min(10,len(agents))))
                for i,a in enumerate(snapshot[:10]):
                    with a.lock:
                        print(f"#{i+1:2d} ID {a.id:3d} score={a.score:6d} games={a.games_played:4d} lines={a.lines_cleared_total:4d} genome=[spd{a.genome.speed} h{a.genome.w_height:.2f} l{a.genome.w_lines:.2f} ho{a.genome.w_holes:.2f} b{a.genome.w_bump:.2f}]")
                # save best genome to best_genome.json (for easy access)
                best = snapshot[0]
                with best.lock:
                    stats = {"score": int(best.score), "games_played": int(best.games_played), "lines": int(best.lines_cleared_total)}
                    save_best_genome(best.genome, stats=stats)
                last_stats = now
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)

# ---------- Rendering ----------
def draw_agent_thumbnail(surf, x, y, w, h, agent: AgentState, rank:int=None, points:int=None, font=None):
    cols = COLS; rows = ROWS
    cell_w = max(2, w // cols)
    cell_h = max(2, h // rows)
    board_w = cell_w * cols; board_h = cell_h * rows
    bx = x + (w - board_w)//2
    by = y + 8
    pygame.draw.rect(surf, COLOR_GRID, (x,y,w,h))
    pygame.draw.rect(surf, COLOR_CELL_BG, (bx-1, by-1, board_w+2, board_h+2))
    with agent.lock:
        grid = agent.board_snapshot if agent.board_snapshot else [[0]*cols for _ in range(rows)]
        cp = agent.current_piece if agent.current_piece else None
        npiece = agent.next_piece
        over = agent.game_over
    # draw cells
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            color = COLORS.get(cell, (120,120,120)) if cell != 0 else (10,10,10)
            rect = pygame.Rect(bx + c*cell_w, by + r*cell_h, cell_w-1, cell_h-1)
            pygame.draw.rect(surf, color, rect)
    # draw current piece overlay
    if cp:
        shape = PIECES[cp['type']][cp['rot']]
        for dr,dc in shape:
            rr = cp['r'] + dr; cc = cp['c'] + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                rect = pygame.Rect(bx + cc*cell_w, by + rr*cell_h, cell_w-1, cell_h-1)
                pygame.draw.rect(surf, COLORS.get(cp['type'], (255,255,255)), rect)
                inner = rect.inflate(-2,-2)
                pygame.draw.rect(surf, (20,20,20), inner)
    # next piece tiny
    if npiece:
        nx = x + w - 44; ny = y + 8
        small = 10
        for dr,dc in PIECES[npiece][0]:
            rect = pygame.Rect(nx + dc*small, ny + dr*small, small-1, small-1)
            pygame.draw.rect(surf, COLORS.get(npiece,(200,200,200)), rect)
    pygame.draw.rect(surf, (40,40,40), (x,y,w,h), 1)
    if font and rank is not None and points is not None:
        badge_text = f"#{rank}  {points}"
        bx2 = x + 6; by2 = y + 6
        bw = max(36, font.size(badge_text)[0] + 8)
        bh = max(16, font.get_linesize())
        pygame.draw.rect(surf, (10,10,10), (bx2-1, by2-1, bw+2, bh+2))
        pygame.draw.rect(surf, (30,30,30), (bx2, by2, bw, bh))
        txt_surf = font.render(badge_text, True, (220,220,220))
        surf.blit(txt_surf, (bx2+4, by2 + (bh - font.get_linesize())//2))

# ---------- Main ----------
def main():
    ensure_genome_dir()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Tetris Population Training — NES Scoring")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14)
    large_font = pygame.font.SysFont("Consolas", 18)

    agents = []
    # create agents, load genomes if available
    for i in range(POPULATION_SIZE):
        loaded = load_genome_file(i)
        if loaded:
            genome = loaded
            print(f"Loaded genome for ID {i} from file.")
        else:
            genome = Genome()
        agents.append(AgentState(id=i, genome=genome))
        # save initial genome file (ensures files exist)
        save_genome_file(agents[-1])

    stop_event = threading.Event()
    manager_thread = threading.Thread(target=evolution_manager, args=(agents, stop_event), daemon=True)
    manager_thread.start()

    thumb_area_w = WINDOW_W - LEADERBOARD_WIDTH - 20
    thumb_area_h = WINDOW_H - 20
    aspect = thumb_area_w / max(1, thumb_area_h)
    cols_thumb = max(1, int(math.ceil(math.sqrt(POPULATION_SIZE * aspect))))
    rows_thumb = int(math.ceil(POPULATION_SIZE / cols_thumb))
    thumb_w = thumb_area_w // cols_thumb
    thumb_h = thumb_area_h // rows_thumb
    VISUAL_LIMIT = POPULATION_SIZE
    visual_indices = list(range(POPULATION_SIZE))

    running = True
    while running and not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False; stop_event.set()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = False; stop_event.set()

        screen.fill(COLOR_BG)
        snapshot_full = sorted(agents, key=lambda a: a.score, reverse=True)
        rank_by_id = {}
        score_by_id = {}
        for idx, agent_obj in enumerate(snapshot_full, start=1):
            rank_by_id[agent_obj.id] = idx
            score_by_id[agent_obj.id] = agent_obj.score

        start_x = 10; start_y = 10
        idx = 0
        for row in range(rows_thumb):
            for col in range(cols_thumb):
                if idx >= VISUAL_LIMIT: break
                agent_idx = visual_indices[idx]
                ax = start_x + col * thumb_w
                ay = start_y + row * thumb_h
                pad = THUMB_MARGIN
                rnk = rank_by_id.get(agent_idx, None)
                pts = score_by_id.get(agent_idx, None)
                draw_agent_thumbnail(screen, ax + pad, ay + pad, thumb_w - pad*2, thumb_h - pad*2, agents[agent_idx], rank=rnk, points=pts, font=font)
                idx += 1
            if idx >= VISUAL_LIMIT: break

        lb_x = WINDOW_W - LEADERBOARD_WIDTH + 10
        lb_y = 10
        pygame.draw.rect(screen, (18,18,18), (WINDOW_W - LEADERBOARD_WIDTH, 0, LEADERBOARD_WIDTH, WINDOW_H))
        title = large_font.render("Leaderboard (Top)", True, (235,235,235))
        screen.blit(title, (lb_x, lb_y))
        y = lb_y + 28
        snapshot = snapshot_full[:20]
        for i,a in enumerate(snapshot):
            with a.lock:
                txt = f"#{i+1:2d} ID{a.id:4d} S{a.score:6d} G{a.games_played:4d} L{a.lines_cleared_total:4d} spd{a.genome.speed}"
            surf = font.render(txt, True, (220,220,220))
            screen.blit(surf, (lb_x, y))
            y += 18
            if y > WINDOW_H - 40: break

        info_lines = [
            f"Population: {POPULATION_SIZE}  Elites: {ELITE_COUNT}  Evolution every {EVOLUTION_INTERVAL_SEC}s",
            f"MutationRate: {MUTATION_RATE}  Press SPACE to stop."
        ]
        iy = WINDOW_H - 60
        for line in info_lines:
            surf = font.render(line, True, (200,200,200))
            screen.blit(surf, (10, iy))
            iy += 16

        pygame.display.flip()
        clock.tick(30)

    # stopping
    stop_event.set()
    manager_thread.join(timeout=2.0)
    # save all genomes on exit
    save_all_genomes(agents)
    pygame.quit()
    print("Stopped. Best genome(s):")
    best = sorted(agents, key=lambda a: a.score, reverse=True)[0]
    print(f"ID {best.id} score={best.score} games={best.games_played} lines={best.lines_cleared_total} genome={best.genome.to_dict()}")
    # also write best genome file one last time
    with best.lock:
        save_best_genome(best.genome, stats={"score": int(best.score), "games_played": int(best.games_played), "lines": int(best.lines_cleared_total)})

if __name__ == "__main__":
    main()
