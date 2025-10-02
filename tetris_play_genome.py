#!/usr/bin/env python3
"""
tetris_play_genome.py

Play-test a single Tetris genome (no evolution). Animated per-frame movement or headless fast runs.

Features:
 - Accepts JSON format:
    { "id": 80, "genome": { "w_height": ..., "w_lines": ..., ... }, "meta": {...} }
   or plain genome dict.
 - --headless : run without rendering (fast).
 - --games N : number of games in headless mode.
 - --silent : suppress per-game prints in headless.
 - --quiet-all : suppress all printing including final summary.
 - --log-file PATH : append per-game CSV rows with genome fields.
 - --show-window : after headless runs, open pygame window showing the final board.
 - --replay : when combined with --show-window, show an animated replay of the last game's snapshots.
"""

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# optional pygame import for visual mode / show-window
try:
    import pygame
except Exception:
    pygame = None

# ---------- Config ----------
ROWS = 20
COLS = 10
CELL = 26
SCREEN_W = COLS * CELL + 260
SCREEN_H = ROWS * CELL + 40

# NES scoring
SCORE_TABLE = {1: 40, 2: 100, 3: 300, 4: 1200}

# Piece definitions
PIECES = {
    'I': [
        [(0,0),(0,1),(0,2),(0,3)],
        [(0,2),(1,2),(2,2),(3,2)],
        [(1,0),(1,1),(1,2),(1,3)],
        [(0,1),(1,1),(2,1),(3,1)]
    ],
    'O': [ [(0,0),(0,1),(1,0),(1,1)] ]*4,
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

COLORS = {
    'I': (0,255,255),
    'O': (255,200,0),
    'T': (160,0,255),
    'S': (0,255,0),
    'Z': (255,0,0),
    'J': (0,0,255),
    'L': (255,140,0),
    0: (12,12,12)
}

# ---------- Utilities ----------
def clamp(v,a,b): return max(a,min(b,v))
def copy_board(b): return [row.copy() for row in b]

def can_place_on_board(board, shape, top_row, left_col):
    rows = len(board); cols = len(board[0])
    for dr,dc in shape:
        r = top_row + dr; c = left_col + dc
        if c < 0 or c >= cols: return False
        if r >= rows: return False
        if r >= 0 and board[r][c] != 0: return False
    return True

def drop_height_on_board(board, shape, left_col):
    rows = len(board)
    for top in range(-6, rows):
        if not can_place_on_board(board, shape, top+1, left_col):
            return top
    return rows-1

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
        out.insert(0, [0]*cols)
    return newb if cleared==0 else out, cleared

def aggregate_height(board):
    rows = len(board); cols = len(board[0])
    heights = []
    for c in range(cols):
        h = 0
        for r in range(rows):
            if board[r][c] != 0:
                h = rows - r
                break
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
        return {'type': piece_type, 'rot': 0, 'r': -2, 'c': (self.cols // 2) - 2}

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

# ---------- Genome dataclass ----------
@dataclass
class Genome:
    w_height: float = -0.5
    w_lines: float = 1.0
    w_holes: float = -1.0
    w_bump: float = -0.5
    speed: int = 6

    def to_dict(self):
        return {
            "w_height": self.w_height,
            "w_lines": self.w_lines,
            "w_holes": self.w_holes,
            "w_bump": self.w_bump,
            "speed": self.speed
        }

    @staticmethod
    def from_dict(d: dict):
        if d is None:
            return Genome()
        def gget(k, alt, default):
            if k in d: return d[k]
            if alt in d: return d[alt]
            return default
        try:
            wh = float(gget("w_height","height",-0.5))
        except Exception:
            wh = -0.5
        try:
            wl = float(gget("w_lines","lines",1.0))
        except Exception:
            wl = 1.0
        try:
            who = float(gget("w_holes","holes",-1.0))
        except Exception:
            who = -1.0
        try:
            wb = float(gget("w_bump","bump",-0.5))
        except Exception:
            wb = -0.5
        try:
            sp = int(gget("speed","spd",6))
        except Exception:
            sp = 6
        return Genome(w_height=wh, w_lines=wl, w_holes=who, w_bump=wb, speed=sp)

# ---------- GenomePlayer (with optional history capture) ----------
class GenomePlayer:
    def __init__(self, genome: Genome, nes_weight: float = 0.02, headless: bool = False, capture_history: bool = False):
        self.genome = genome
        self.nes_weight = nes_weight
        self.headless = headless
        self.capture_history = capture_history
        self.model = TetrisModel()
        self.bag = []
        self.current = self.model.spawn_piece(self._draw_piece())
        self.next_piece = self._draw_piece()
        self.cumulative_score = 0

        now = time.time()
        self.plan = None
        self.last_action_time = now - 10.0
        self.last_fall_time = now - 10.0
        self.lock_on_ground = False
        self.lock_start = None

        # history captures board snapshots after each lock (for replay). only used if capture_history True.
        self.history: List[List[List[int]]] = []
        # capture initial empty board state if requested
        if self.capture_history:
            self.history.append(copy_board(self.model.grid))

    def _draw_piece(self):
        if not self.bag:
            self.bag = PIECE_KEYS.copy()
            random.shuffle(self.bag)
        return self.bag.pop()

    def action_delay(self):
        if self.headless:
            return 0.0
        s = clamp(self.genome.speed, 1, 10)
        return max(0.01, 0.04 * (1.0 - (s - 1)/9.0))

    def fall_delay(self):
        if self.headless:
            return 0.0
        s = clamp(self.genome.speed, 1, 10)
        return ((11 - s) / 10.0) * 0.9

    def compute_best_plan(self):
        best_val = -1e12
        best_plan = None
        ptype = self.current['type']
        for rot_idx in range(len(PIECES[ptype])):
            shape = PIECES[ptype][rot_idx]
            for left in range(-4, COLS+1):
                top = drop_height_on_board(self.model.grid, shape, left)
                if not can_place_on_board(self.model.grid, shape, top, left):
                    continue
                newb, cleared = apply_shape_and_clear(self.model.grid, shape, top, left, piece_id=9)
                agg, heights = aggregate_height(newb)
                holes = count_holes(newb)
                bump = bumpiness_from_heights(heights)
                g = self.genome
                heuristic = g.w_height * agg + g.w_lines * cleared + g.w_holes * holes + g.w_bump * bump
                nes_score = SCORE_TABLE.get(cleared, 0) * (self.model.level + 1)
                val = heuristic + self.nes_weight * nes_score
                val += -abs((left+2) - (COLS//2)) * 0.02
                if val > best_val:
                    best_val = val
                    best_plan = (rot_idx, left, top, cleared)
        self.plan = None if best_plan is None else {'rot': best_plan[0], 'col': best_plan[1], 'top': best_plan[2], 'cleared': best_plan[3]}

    def step(self, stop_event=None) -> bool:
        """Advance one micro-step. Returns True if still running."""
        if self.model.game_over:
            return False
        now = time.time()
        if self.plan is None:
            self.compute_best_plan()
            if self.plan is None:
                self.model.game_over = True
                return False
        do_action = (now - self.last_action_time) >= self.action_delay()
        do_fall = (now - self.last_fall_time) >= self.fall_delay()

        # rotation
        if do_action and (self.current['rot'] % len(PIECES[self.current['type']])) != (self.plan['rot'] % len(PIECES[self.current['type']])):
            new_rot = (self.current['rot'] + 1) % len(PIECES[self.current['type']])
            test_piece = {'type': self.current['type'], 'rot': new_rot, 'r': self.current['r'], 'c': self.current['c']}
            if self.model.can_place(test_piece):
                self.current['rot'] = new_rot
            self.last_action_time = now
            return True

        # horizontal
        if do_action and self.current['c'] != self.plan['col']:
            if self.current['c'] < self.plan['col']:
                cand = {'type': self.current['type'], 'rot': self.current['rot'], 'r': self.current['r'], 'c': self.current['c']+1}
                if self.model.can_place(cand):
                    self.current['c'] += 1
            elif self.current['c'] > self.plan['col']:
                cand = {'type': self.current['type'], 'rot': self.current['rot'], 'r': self.current['r'], 'c': self.current['c']-1}
                if self.model.can_place(cand):
                    self.current['c'] -= 1
            self.last_action_time = now
            return True

        # falling / lock
        if do_fall:
            cand = {'type': self.current['type'], 'rot': self.current['rot'], 'r': self.current['r']+1, 'c': self.current['c']}
            if self.model.can_place(cand):
                self.current['r'] += 1
                self.last_fall_time = now
                self.lock_on_ground = False
                self.lock_start = None
                return True
            else:
                if not self.lock_on_ground:
                    self.lock_on_ground = True
                    self.lock_start = now
                    self.last_fall_time = now
                    return True
                else:
                    if (now - self.lock_start) >= 0.12:
                        cleared_lines = self.model.lock_piece(self.current)
                        nes_add = SCORE_TABLE.get(cleared_lines, 0) * (self.model.level + 1)
                        self.cumulative_score += nes_add
                        # capture history snapshot if requested (copy the grid after lock)
                        if self.capture_history:
                            self.history.append(copy_board(self.model.grid))
                        # spawn next
                        self.current = self.model.spawn_piece(self.next_piece)
                        self.next_piece = self._draw_piece()
                        self.plan = None
                        self.lock_on_ground = False
                        self.lock_start = None
                        self.last_action_time = now
                        self.last_fall_time = now
                        return not self.model.game_over

        return not self.model.game_over

# ---------- Rendering helpers ----------
def render(screen, player: GenomePlayer, genome: Genome, font):
    screen.fill((18,18,18))
    bx = 12; by = 12
    # draw grid
    for r in range(ROWS):
        for c in range(COLS):
            val = player.model.grid[r][c]
            color = COLORS.get(val, (12,12,12)) if val != 0 else (8,8,8)
            rect = pygame.Rect(bx + c*CELL, by + r*CELL, CELL-1, CELL-1)
            pygame.draw.rect(screen, color, rect)
    # draw current falling piece (animated position)
    cp = player.current
    if cp:
        for dr,dc in PIECES[cp['type']][cp['rot']]:
            rr = cp['r'] + dr; cc = cp['c'] + dc
            if 0 <= rr < ROWS and 0 <= cc < COLS:
                rect = pygame.Rect(bx + cc*CELL, by + rr*CELL, CELL-1, CELL-1)
                pygame.draw.rect(screen, COLORS.get(cp['type'], (255,255,255)), rect)
                inner = rect.inflate(-4,-4)
                pygame.draw.rect(screen, (18,18,18), inner)
    # draw next piece and info
    nx = bx + COLS*CELL + 22
    ny = by
    if pygame:
        label = font.render("Next:", True, (220,220,220))
        screen.blit(label, (nx, ny))
        ny += 22
    if player.next_piece:
        for dr,dc in PIECES[player.next_piece][0]:
            rect = pygame.Rect(nx + dc*18, ny + dr*18, 16, 16)
            pygame.draw.rect(screen, COLORS.get(player.next_piece,(200,200,200)), rect)
    ny += 70
    # genome - exact values, no rounding
    genome_info = json.dumps(genome.to_dict(), indent=None)
    stat_lines = [
        f"Score (NES): {player.cumulative_score}",
        f"Lines: {player.model.lines_cleared}",
        f"Level: {player.model.level}",
        f"Genome: {genome_info}",
        "",
        "Press ESC to quit. Press P to pause."
    ]
    for i, ln in enumerate(stat_lines):
        surf = font.render(ln, True, (230,230,230))
        screen.blit(surf, (nx, ny + i*20))
    pygame.display.flip()

# ---------- I/O helpers ----------
def load_genome_from_file(path: str) -> Genome:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "genome" in payload and isinstance(payload["genome"], dict):
        gdict = payload["genome"]
    else:
        gdict = payload
    return Genome.from_dict(gdict)

def append_csv_row(path: str, row: List, header: Optional[List[str]] = None):
    write_header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header and header:
            w.writerow(header)
        w.writerow(row)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Play a single Tetris genome (no evolution).")
    parser.add_argument('--genome', type=str, default=None, help='Inline JSON genome.')
    parser.add_argument('--genome-file', type=str, default=None, help='Path to JSON file with genome (wrapper or plain).')
    parser.add_argument('--nes-weight', type=float, default=0.02, help='Weight applied to NES score when evaluating moves.')
    parser.add_argument('--headless', action='store_true', help='Run without rendering; runs games fast (no animation delays).')
    parser.add_argument('--games', type=int, default=1, help='Number of games to run in headless mode (default 1).')
    parser.add_argument('--silent', action='store_true', help='In headless mode, suppress per-game text output.')
    parser.add_argument('--quiet-all', action='store_true', help='Suppress all console output (including final summary).')
    parser.add_argument('--log-file', type=str, default=None, help='Optional CSV file to append per-game stats.')
    parser.add_argument('--show-window', action='store_true', help='After headless games, show a pygame window with last game final board.')
    parser.add_argument('--replay', action='store_true', help='If --show-window, animate a replay of the last headless game (if available).')
    args = parser.parse_args()

    # load genome
    genome: Genome
    if args.genome_file:
        try:
            genome = load_genome_from_file(args.genome_file)
        except Exception as e:
            print("Failed to load genome file:", e)
            return
    elif args.genome:
        try:
            data = json.loads(args.genome)
            if isinstance(data, dict) and "genome" in data and isinstance(data["genome"], dict):
                genome = Genome.from_dict(data["genome"])
            else:
                genome = Genome.from_dict(data)
        except Exception as e:
            print("Failed to parse genome JSON:", e)
            return
    else:
        genome = Genome()
        if not args.quiet_all:
            print("No genome provided — using default genome.")

    if not args.quiet_all:
        print("Testing genome (exact values):")
        print(json.dumps(genome.to_dict(), indent=2))

    # HEADLESS MODE
    if args.headless:
        total_score = 0
        total_lines = 0
        total_games = max(1, int(args.games))
        start_time = time.time()
        last_player: Optional[GenomePlayer] = None
        csv_header = ["game_index", "score", "lines", "elapsed_s", "w_height", "w_lines", "w_holes", "w_bump", "speed"]

        for i in range(total_games):
            capture = bool(args.replay and i == (total_games - 1))
            player = GenomePlayer(genome, nes_weight=args.nes_weight, headless=True, capture_history=capture)
            t0 = time.time()
            # run until game over (tight loop)
            while player.step():
                continue
            t_elapsed = time.time() - t0
            total_score += player.cumulative_score
            total_lines += player.model.lines_cleared
            last_player = player
            if not args.silent and not args.quiet_all:
                print(f"Game {i+1}/{total_games} finished — Score: {player.cumulative_score}, Lines: {player.model.lines_cleared}, Time: {t_elapsed:.6f}s")
            if args.log_file:
                row = [
                    i+1,
                    player.cumulative_score,
                    player.model.lines_cleared,
                    f"{t_elapsed:.6f}",
                    genome.w_height,
                    genome.w_lines,
                    genome.w_holes,
                    genome.w_bump,
                    genome.speed
                ]
                append_csv_row(args.log_file, row, header=csv_header)

        elapsed = time.time() - start_time
        if not args.quiet_all:
            print("=== Headless summary ===")
            print(f"Games: {total_games}, Total score: {total_score}, Avg score: {total_score/total_games:.6f}")
            print(f"Total lines: {total_lines}, Avg lines: {total_lines/total_games:.6f}")
            print(f"Elapsed time: {elapsed:.3f}s, Games/sec: {total_games/elapsed:.2f}")

        # optionally show final board or replay
        if args.show_window and last_player is not None:
            if pygame is None:
                print("Pygame not available; cannot show window.")
                return
            pygame.init()
            screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("Tetris — Last headless game (final board/replay)")
            font = pygame.font.SysFont("Consolas", 16)
            clock = pygame.time.Clock()
            if args.replay and last_player.capture_history and len(last_player.history) > 0:
                # animate snapshots (each snapshot is a grid after locks)
                snapshots: List[List[List[int]]] = last_player.history
                # show each snapshot for a short time, repeat until closed
                showing = True
                fps = 30
                frame_hold = 0.4  # seconds per snapshot
                while showing:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            showing = False; break
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                            showing = False; break
                    for snap in snapshots:
                        # render snap
                        screen.fill((18,18,18))
                        bx = 12; by = 12
                        for r in range(ROWS):
                            for c in range(COLS):
                                v = snap[r][c]
                                color = COLORS.get(v, (12,12,12)) if v != 0 else (8,8,8)
                                rect = pygame.Rect(bx + c*CELL, by + r*CELL, CELL-1, CELL-1)
                                pygame.draw.rect(screen, color, rect)
                        # info on right
                        nx = bx + COLS*CELL + 22
                        ny = by + 10
                        lines_txt = font.render(f"Score (NES): {last_player.cumulative_score}", True, (230,230,230))
                        screen.blit(lines_txt, (nx, ny))
                        pygame.display.flip()
                        t_end = time.time() + frame_hold
                        while time.time() < t_end:
                            for ev in pygame.event.get():
                                if ev.type == pygame.QUIT:
                                    showing = False; break
                                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                                    showing = False; break
                            clock.tick(fps)
                        if not showing:
                            break
                pygame.quit()
            else:
                # show final board static
                showing = True
                while showing:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            showing = False; break
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                            showing = False; break
                    render(screen, last_player, genome, font)
                    clock.tick(30)
                pygame.quit()

        return

    # VISUAL MODE - require pygame
    if pygame is None:
        print("Pygame is required for visual mode. Install pygame or use --headless.")
        return

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Tetris — Play Genome (animated)")
    font = pygame.font.SysFont("Consolas", 16)
    clock = pygame.time.Clock()

    player = GenomePlayer(genome, nes_weight=args.nes_weight, headless=False, capture_history=False)
    running = True
    paused = False

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False; break
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False; break
                if ev.key == pygame.K_p:
                    paused = not paused

        if not paused:
            alive = player.step()
            if not alive:
                if not args.quiet_all:
                    print("GAME OVER — Score (NES):", player.cumulative_score, "Lines:", player.model.lines_cleared)
                # restart automatically
                player = GenomePlayer(genome, nes_weight=args.nes_weight, headless=False, capture_history=False)
                time.sleep(0.35)

        render(screen, player, genome, font)
        clock.tick(30)

    pygame.quit()
    if not args.quiet_all:
        print("Exited. Last game score:", player.cumulative_score)

if __name__ == "__main__":
    main()
