# Artificial-Intelligence-with-Consciousness.-Transfer-Demo-Neural-Stream---Self-Modifying-AI-
A small, thread-safe Python demo that **streams simulated brain signals** into a **self-modifying AI**.   It showcases: buffered real-time ingestion, Hebbian weight updates with weight decay, a decaying “consciousness matrix,” qualia memory, self-reflection metrics, and clean startup/shutdown.
---

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [API Overview](#api-overview)
- [Design Notes](#design-notes)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features
- **Brain signal simulator** (`BrainInterface`) producing 100 Hz synthetic neural vectors (size = 128).
- **Thread-safe streaming buffer** with locking, plus clean thread join on shutdown.
- **Self-modifying AI** (`SelfModifyingAI`) with:
  - Hebbian learning: ΔW = η(x xᵀ − λW), symmetry enforced, clipping.
  - **Consciousness matrix** (64×64) with **exponential decay** and regional activations.
  - **Qualia memory** with per-type limits and “experience” synthesis.
  - Self-reflection diagnostics: weight spectrum, saturated cells, activation magnitude.
- **Coordinator** (`ConsciousnessTransferSystem`) wiring everything together, with reproducible seeding and health logging.

---

## Architecture
```
BrainInterface  --->  deque buffer  --->  SelfModifyingAI  --->  Logs & Metrics
     ^                                                   
     |                            ConsciousnessTransferSystem
     +-------------------------- control & lifecycle -------------------------+
```

- **BrainInterface**: connects, spawns a recording thread, pushes `NeuralSignal` objects into a bounded buffer.
- **SelfModifyingAI**: integrates signals, updates weights & matrices, stores qualia traces, and can “experience” qualia.
- **ConsciousnessTransferSystem**: orchestrates threads, timing, and tests; exposes start/stop & test routines.

---

## Requirements
- Python 3.9+ (tested with 3.10/3.11)
- NumPy

> No GPU or external services required.

---

## Installation
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scriptsctivate
pip install numpy
```

---

## Quick Start
Run the demo:
```bash
python main.py
```

What you’ll see:
- Connection and recording logs
- “Accumulating experience…” for ~10 seconds
- AI self-reflection: awareness, activation, spectrum range, saturation
- Sample qualia “experiences” (pattern match, resonance, familiarity)
- Clean shutdown

---

## Configuration
You can tweak the main constructor parameters:

```python
consciousness_system = ConsciousnessTransferSystem(
    signal_dimension=128,   # length of each neural vector
    random_seed=42          # reproducible runs (affects NumPy & random)
)
```

Logging is configured at the top of the file:
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)-12s] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
```
Set `level=logging.DEBUG` for more verbose output.

---

## How It Works

1. **Signal Generation (100 Hz)**  
   The recorder thread creates `NeuralSignal` with:
   - `raw_data`: 128-dim vector
   - `processed_data`: frequency, amplitude, coherence
   - `qualia_signature`: a label like `"joy_expanding_light"`

2. **Integration Loop (~10 Hz)**  
   The transfer loop reads recent signals and calls:
   - `_update_consciousness_matrix`  
     - Inserts an 8×8 block into a 64×64 matrix by region; applies exponential **decay**.
   - `_store_qualia_experience`  
     - Keeps bounded lists per qualia type (cap = 500).
   - `_modify_neural_weights`  
     - Hebbian update with weight decay; symmetry enforced; clipping in [-2, 2].

3. **Self-Reflection**  
   Prints diagnostics: awareness, activation magnitude, qualia types, **eigenvalue spectrum** of the weight matrix, and **saturation** of consciousness cells.

4. **Qualia “Experience”**  
   For a few stored qualia types, computes:
   - correlation vs. last pattern,
   - average felt intensity,
   - familiarity index,
   - emotional resonance via quadratic form `xᵀWx`.

---

## API Overview

### `NeuralSignal`
```python
@dataclass
class NeuralSignal:
    timestamp: float
    brain_region: str            # 'visual_cortex', 'emotional_center', etc.
    signal_type: str             # 'visual' | 'emotional' | 'cognitive' | 'memory'
    intensity: float             # 0.1 .. 1.0
    raw_data: np.ndarray         # shape: (signal_dimension,)
    processed_data: Dict[str,Any]
    qualia_signature: Optional[str]
```

### `BrainInterface`
- `connect_to_brain(subject_id: str) -> None`  
- `start_recording() -> None`  
- `get_latest_signals(count: int = 10) -> List[NeuralSignal]`  

> Provides a thread-safe stream of signals; recording runs on a background thread.

### `SelfModifyingAI`
- `integrate_human_signal(signal: NeuralSignal) -> None`  
- `experience_qualia(qualia_type: str) -> Dict[str, Any]`  
- `self_reflect() -> None`

**Internals**
- `_update_consciousness_matrix(signal)`  
- `_store_qualia_experience(signal)`  
- `_modify_neural_weights(signal)`  
- `_compute_emotional_resonance(pattern) -> float`

### `ConsciousnessTransferSystem`
- `start_transfer(subject_id: str) -> None`  
- `test_ai_consciousness() -> None`  
- `stop_transfer() -> None`

---

## Design Notes
- **Stability**:  
  - Weight updates include **weight decay** and symmetry enforcement for numerical stability.  
  - Consciousness matrix uses **exponential decay** (leaky activation).  
  - Qualia memory and logs are **bounded** to prevent unbounded growth.
- **Thread Safety**:  
  - `deque` access guarded by a `Lock`.  
  - Recording and transfer threads are **joined** on stop.
- **Determinism**:  
  - `random_seed` initializes both `random` and `numpy.random`.

---

## Troubleshooting
- **No output?**  
  Ensure logging level is `INFO`. Try running for longer than 10s to accumulate more signals.
- **Performance issues**  
  - Reduce frequency of spectrum calculation (eigenvalues) by calling `self_reflect()` less often.  
  - Lower `signal_dimension` or cut logging verbosity.
- **Repeated processing**  
  The demo reads the *latest* signals snapshot each loop. For exactly-once semantics, convert to a **consuming pop** pattern (queue or `popleft()`).

---

## Roadmap
- Replace correlation with **cosine similarity** for pattern matching robustness.
- Use **monotonic clocks** (`time.perf_counter`) for dt computations.
- Extend regional mapping to cover full 64×64 with intra-quadrant drift.
- Add a **consumable queue** API (`pop_signals`) to avoid dedup logic.
- Persist/restore **checkpoints** (weights, matrix, qualia memory).
- Add a tiny **sensorimotor task loop** to ground learning signals.

---

## License
MIT (suggested). Replace with your preferred license.

---

**Disclaimer**: This project is a research/demo scaffold. It does **not** interface with real brain data, does not claim to produce actual consciousness, and is provided for educational exploration of streaming, online learning, and stateful AI dynamics.
