# RL-PCG Corridors (SeGMaN-Style Grid Generator)

Procedural grid layout generation using **Reinforcement Learning (RL)** with a **PPO agent** and a **custom CNN policy**.  
The agent learns to construct solvable levels with meaningful corridor-like wall patterns by sequentially editing a 2D grid.

----------
- Requires Python 3.8 (Higher versions haven't been tested, but can work.)
----------
## 🧩 Overview
<table><tr><td><img src="/eval_v3_stoch/keepers/ep_00616.png" width="200" height="200"></td><td><img src="/eval_v3_stoch/keepers/ep_00852.png" width="200" height="200"></td><td><img src="/eval_v3_stoch/keepers/ep_01504.png" width="200" height="200"></td></tr></table>
<img src="/montage_stoch.png" width="600" height="600">


This project explores **Procedural Content Generation via Reinforcement Learning (PCGRL)** on a minimal **Grid-based Level Generation** task inspired by SeGMaN’s compositional planning ideas.

-   **Goal:** Train an agent to design 13×13 maps containing:
    
    -   A **Robot (R)**
        
    -   An **Object (O)**
        
    -   A **Goal (G)**
        
    -   A structured layout of **Walls (W)** forming navigable corridors.
        
-   **Environment:** `GridPCGEnv`
    
    -   Turn-based editing process (each step modifies a single cell)
        
    -   Fixed horizon (default `max_steps=192`)
        
    -   Reward favors _solvability_ (R→O→G paths) and _wall structure quality_.
        
----------

## 🏗️ Environment: `grid_pcg_env.py`

### Entities and Grid Encoding

| Symbol | Meaning | Channel |
| :--- | :--- | :--- |
| 0 | Empty | `plane[0]` |
| 1 | Wall | `plane[1]` |
| 2 | Robot | `plane[2]` |
| 3 | Object | `plane[3]` |
| 4 | Goal | `plane[4]` |

Observation: **(H, W, 5)** one-hot tensor  
Action space: **Discrete(H × W × 5)** — choose a cell + tile type.

----------

### Episode Flow

1.  Agent starts with an empty grid or partial “curriculum” placement (0–2 entities).
    
2.  Each action modifies one cell.
    
3.  Episode terminates after `max_steps` edits.
    
4.  Terminal reward computed on the final grid.
    

----------

### Reward Components

| Term | Description | Effect |
| :--- | :--- | :--- |
| ✅ **Validity** | Must have 1 × Robot, 1 × Object, 1 × Goal | Invalid layouts → −1.0 |
| ✅ **Solvability** | Shortest-path BFS R→O and O→G | Missing path → −0.5 |
| ➕ **Path length (L1 + L2)** | Encourages non-trivial maps | +α · (L1 + L2) |
| ⚖️ **Wall ratio target** | Penalize deviation from 0.25 | −β · |
| 🧱 **Corridor quality** | +λ₁ · adj_per_wall − λ₂ · iso_frac − λ₃ · block_2×2 | Rewards long chains, penalizes blobs |
| 🚫 **Entity adjacency** | Penalize touching R–O or O–G | −γ |
| 🚧 **Border penalty** | Entities near edges | −δ |
| ✳️ **Step shaping (small)** | Bonus if adjacency↑, iso↓ | Smoothes learning |

----------

## 🧠 PPO Training: `train_ppo.py`

### Framework

-   **Algorithm:** PPO (Stable-Baselines3)
    
-   **Policy:** Custom **CNN** (`small_cnn.py`)
    
-   **Parallelism:** 8 vectorized envs (DummyVecEnv)
    
-   **Hardware:** CUDA-enabled GPU
    

### Training, Evaluation, Bulk Sampling Commands in Order

```bash
python train_ppo.py \
  --size 13 --max_steps 192 --n_envs 8 \
  --total_timesteps 1_500_000 \
  --n_steps 1024 --batch_size 4096 \
  --lr 3e-4 --ent_coef 0.03 --use_curriculum \
  --logdir runs/ppo_grid_corridors_v3

python eval_model.py --model runs/ppo_grid_corridors_v3/ppo_grid_nomask_final.zip \
  --episodes 256 --max_steps 192

python eval_and_select.py \
  --model runs/ppo_grid_corridors_v3/ppo_grid_nomask_final.zip \
  --episodes 2000 --max_steps 192 \
  --w_min 0.20 --w_max 0.35 --adj_min 0.15 --iso_max 0.30 --min_Lsum 14 \
  --out_dir eval_v3_stoch --use_curriculum

```
----------

## 🧠 Understanding Wall Behavior

The agent learns a **“safe corridor” policy**:

-   Adds walls while preserving paths.
    
-   Stops increasing density when extra walls risk invalidating solvability.  
    Hence mean wall ratio stabilizes below 0.1, though samples can reach 0.18–0.20.
    

For production, stochastic sampling + filtering is preferred:  
the best 10–20 % of rollouts match design criteria perfectly.

----------

## 🗃️ Repository Structure

```
rlpcg-4-segman/
│
├── grid_pcg_env.py           # Environment definition
├── small_cnn.py              # Custom CNN policy
├── train_ppo.py              # PPO training script
├── eval_model.py             # Basic evaluation
├── eval_and_select.py        # Bulk sampling + filtering
├── montage_sample.py         # Visual montage creation
├── runs/ppo_grid_corridors_v3/
│   └── ppo_grid_nomask_final.zip   # Final model (LFS)
│   └── best_model.zip   # Best model
│   └── ppo_ckpt_404800_steps.zip   # Usable model
│   └── ppo_ckpt_1011008_steps.zip   # Usable model
├── montage_stoch.png
├── check_env.py
├── .gitignore
├── .gitattributes
└── README.md

```

----------

## 💾 Model Management (Git LFS)

Large model weights are stored via **Git LFS**:

```bash
git lfs install
git lfs track "runs/ppo_grid_corridors_v3/ppo_grid_nomask_final.zip"
git add .gitattributes
```

This keeps repository size small while preserving reproducibility.

----------

## 🔬 Interpretation Summary

> **“After 1.5 M steps, the PPO agent reached a stable corridor-forming behavior.  
> Explained variance ≈ 0.9, KL ≈ 0.015, entropy ≈ −5.5 indicate balanced exploration and convergence.  
> Generated levels are solvable (> 0.9 valid), with distinct corridor structures and minimal isolated walls.  
> Remaining variability reflects alternate valid building policies rather than instability.”**

----------

## 🧭 Next Steps

1.  Extend horizon (`max_steps = 256`) for richer edits.
    
2.  Slightly increase wall-density reward to reach 0.15–0.25 ratio.
    
3.  Automate dataset building using `eval_and_select.py` for reproducible level sets.
    
4.  (Optional) Fine-tune low-entropy model for deterministic generation.

5. **Transition to .g files and SeGMaN for reward signals.** 

6. Upon 5th step's success -> Transition to 3D space  
    

----------

## 📚 Acknowledgments

-   Environment design and inspiration from **PCGRL**  (Khalifa et al., 2020).
-   Idea of using RL to generate PCG for SeGMaN from **Solving Sequential Manipulation Puzzles by Finding Easier Subproblems** (Levit et al., 2024).
-   Reinforcement-learning backbone: **Stable-Baselines3 (PPO)**.
    
-   Project conducted under Bilkent University LiRA Lab research (2025).
    

----------
