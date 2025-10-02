# Tetris AI — Genetic / Heuristic Player

This project trains and tests a **genetic / heuristic-based Tetris AI**. It contains two main scripts:

* `tetris.py` — a trainer that evolves genomes (heuristic weight sets).
* `tetris_play_genome.py` — a tester that plays a single genome (visual or headless).

Genomes are stored as JSON files. Example genomes are provided in the `genomes/` directory.

---

## 📂 Repository Layout

```
.
├── genomes/                     # Example genomes (JSON). Users can contribute via PRs.
├── tetris.py                    # Evolutionary trainer
├── tetris_play_genome.py        # Play / test a single genome
└── README.md                    # This file
```

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/entity12208/Tetris_AI.git
cd Tetris_AI
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install pygame
```

### 3. Train genomes

```bash
python tetris.py
```

* Adjust training parameters directly in `tetris.py`.
* New genomes will be generated and improved over time.

### 4. Test a genome

```bash
python tetris_play_genome.py --genome-file genomes/best.json
```

---

## 🧬 Genome Format

Genomes are JSON objects describing heuristic weights and speed:

```json
{
  "id": 80,
  "genome": {
    "w_height": -0.62,
    "w_lines": 1.02,
    "w_holes": -1.24,
    "w_bump": -0.24,
    "speed": 9
  },
  "meta": {
    "score": 0,
    "games_played": 0,
    "lines_cleared_total": 0,
    "updated": "2025-10-02T18:16:33Z"
  }
}
```

* `genome` contains the actual weights used for evaluation.
* `id` and `meta` are optional metadata produced by the trainer.
* `tetris_play_genome.py` accepts both wrapper-format JSONs and plain genome dicts.

---

## 🎮 Player Script Usage

`tetris_play_genome.py` supports multiple options for running genomes:

```
--genome         : Inline JSON genome (string)
--genome-file    : Path to a genome JSON file
--nes-weight     : Weight applied to NES scoring bonus (default 0.02)
--headless       : Run without rendering (faster, batch mode)
--games N        : Number of games to run (default: 1)
--silent         : Suppress per-game prints in headless mode
--quiet-all      : Suppress all console output
--log-file PATH  : Save per-game results as CSV
--show-window    : Show final game board (after headless runs)
--replay         : Animate last game replay when using --show-window
```

### Examples

Run with visual window:

```bash
python tetris_play_genome.py --genome-file genomes/best.json
```

Headless 100 runs, log results, no console output:

```bash
python tetris_play_genome.py --genome-file genomes/best.json --headless --games 100 --log-file results.csv --quiet-all
```

Replay the last headless game visually:

```bash
python tetris_play_genome.py --genome-file genomes/best.json --headless --games 10 --replay --show-window
```

---

## 📊 Scoring & Fitness

* **NES-style scoring** is used for line clears:

  * 1 line → 40
  * 2 lines → 100
  * 3 lines → 300
  * 4 lines → 1200

* **Heuristic features**:

  * Aggregate height
  * Lines cleared
  * Holes
  * Bumpiness

* **Fitness** is a combination of heuristic evaluation and NES scoring (scaling via `--nes-weight`).

---

## ⚡ Tips to Improve Training Speed

* Increase **population size** for more diversity.
* Run **more games per genome** to reduce randomness.
* Tune the **mutation rate** (higher = more exploration, lower = stability).
* Use **elite retention** to keep top genomes each generation.
* Train in **headless mode** to speed up iteration.

---

## 🤝 Contributing

* Example genomes are in the `genomes/` directory. You can contribute your own via **Pull Requests**.
* If you encounter bugs, feature requests, or ideas, open an **Issue** in GitHub.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.