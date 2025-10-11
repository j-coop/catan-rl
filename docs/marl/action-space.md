# 🧩 Catan MARL Environment — Action Space Specification

This document defines the **discrete action space** used in the `CatanEnv` PettingZoo environment for multi-agent reinforcement learning (MARL).  
It outlines how player decisions are represented as discrete actions, and explains the design rationale behind each action group.

---

## 🎯 Design Goals

- **Finer control granularity:**  
  Each environment step represents one *atomic player action* (e.g., build, trade, or play a card).
  
- **Learning clarity:**  
  Avoid cluttering the action space with redundant or low-value actions.

- **Realistic gameplay flow:**  
  Maintain Catan’s natural sequence of decisions and events, while simplifying asynchronous phases (e.g., discarding).

---

## 🧠 Action Space Overview

| Category | Count | Description |
|-----------|--------|-------------|
| Build Settlement | `N_NODES` | Place a settlement on a valid, unoccupied node. |
| Build City | `N_NODES` | Upgrade an existing settlement on a node to a city. |
| Build Road | `N_EDGES` | Place a road on a valid edge connected to owned structures. |
| Buy Development Card | `1` | Purchase one card from the development deck. |
| Play Development Card | `5` | Play a specific development card type (Knight, Monopoly, Road Building, Year of Plenty, Victory Point). |
| Move Robber | `19` | Move the robber to a new tile (stealing handled heuristically). |
| Trade with Bank / Ports | `20` | Execute a trade of one resource for another at the best available rate for the player. |
| End Turn | `1` | Explicitly end the current player’s turn. |
| **Total (standard board)** | ≈ **226** | For `N_NODES=54`, `N_EDGES=72`. |

---

## ⚙️ Detailed Breakdown

### 🏠 Build Actions

| Action | Count | Description |
|--------|--------|-------------|
| **Build Settlement** | `N_NODES` | Each node corresponds to one possible settlement location. Only valid if player can afford and location is free. |
| **Build City** | `N_NODES` | Each node corresponds to one possible upgrade location (requires existing settlement). |
| **Build Road** | `N_EDGES` | Each edge corresponds to one possible road placement connected to owned infrastructure. |

> These actions are masked dynamically during gameplay to ensure only valid placements are available.

---

### 🧾 Development Cards

| Action | Count | Description |
|--------|--------|-------------|
| **Buy Dev Card** | 1 | Purchase one card from the deck if affordable. |
| **Play Dev Card** | 5 | One action per card type: Knight, Monopoly, Road Building, Year of Plenty, Victory Point. |

---

### 🪓 Robber Movement

| Action | Count | Description |
|--------|--------|-------------|
| **Move Robber** | 19 | One action per tile index. Stealing target is chosen heuristically based on opponents adjacent to that tile (e.g., most resources). |

> Keeping the robber as 19 actions maintains spatial learning while avoiding an explosion in action space size.

---

### 💱 Trading

| Action | Count | Description |
|--------|--------|-------------|
| **Trade with Bank / Ports** | 20 | Encodes all (give_resource → get_resource) pairs. The trade is executed using the best available ratio (port or 4:1). |

> This unified design includes both standard 4:1 bank trades and improved port trades, without redundant actions for worse ratios.

---

### ⏭️ Turn Management

| Action | Count | Description |
|--------|--------|-------------|
| **End Turn** | 1 | Completes the player’s turn. Next player’s turn begins with an automatic dice roll. |

---

## 🚫 Excluded Actions (Heuristic / Automatic)

| Action | Reason for Exclusion | Handling                                                                                           |
|--------|----------------------|----------------------------------------------------------------------------------------------------|
| **Trading with Players** | Non-stationary interaction complexity. | Omitted (may be added in later version).                                                           |
| **Discarding on 7** | Asynchronous event for multiple players. | Handled via heuristic (e.g., discard least valuable resources) - should be added in later version. |
| **Claiming Longest Road / Largest Army** | Deterministic, state-driven effects. | Automatically updated in game logic.                                                               |
| **Initial Placement Phase** | Unique rules and reversed order. | Handled by separate setup model before main gameplay.                                              |

---

## 🧮 Summary Formula

The total discrete action space size is:

ActionSpaceSize = 2 * N_nodes + N_edges + 1 + 5 + 19 + 20 + 1

Where:
- `N_nodes` = number of buildable intersections (typically 54)
- `N_edges` = number of buildable edges (typically 72)

Total (standard board):

ActionSpaceSize = 2 * 54 + 72 + 1 + 5 + 19 + 20 + 1 = 226 actions

---

## 🧭 Future Extensions

| Potential Feature | Description |
|-------------------|-------------|
| Player-to-player trades | Introduce negotiation and dynamic offers between agents. |
| Discard phase as separate loop | Allow each agent to act asynchronously during 7-roll discard events. |
| Robber target selection | Expand move robber to include explicit “choose player to steal from.” |
| Dynamic action grouping | Use hierarchical policies to manage separate sub-action spaces (build / trade / dev card). |

---

_Last updated: 2025-10-11_
