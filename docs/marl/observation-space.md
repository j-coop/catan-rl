# 🧩 Catan Multi-Agent Observation Space Specification

This document describes the observation vector architecture used by each agent in the multi-agent Catan environment.

Each agent receives a **fixed-length observation vector** composed of three sections:

1. **Global board state**
2. **Self (current player) information**
3. **Other players' public information**

The observation is **rotated per agent**, so that:
- The observing agent always appears as *Player 1 (self)*.
- Other players are ordered cyclically: next, next, and previous.

---

## 1️⃣ Global Board State

Encodes all publicly visible game elements: tiles, ports, roads, nodes, and robber.

| Subcomponent | Count | Feature breakdown | Description |
|---------------|--------|-------------------|--------------|
| **Tiles (19)** | 19 × 8 = 152 | - 6 one-hot resource types (wood, brick, wheat, sheep, ore, desert) <br> - 1 normalized number token (`2–12 → value/12`) <br> - 1 robber flag (`1` if robber here or number=7) | Resource and production info per tile |
| **Ports (9)** | 9 × 11 = 99 | - 6 one-hot port types (5 specific + 1 generic 3:1) <br> - 5 one-hot owner (4 players + empty) | Represents trade opportunities and control |
| **Roads (72)** | 72 × 5 = 360 | - 5 one-hot ownership (4 players + empty) | Tracks all built roads |
| **Nodes (54)** | 54 × 6 = 324 | - 4 one-hot ownership (players + empty) <br> - 2-hot building type (settlement, city) | Represents building distribution |
| **Total (Global)** | **998 + 99 - overlap adjustments → 935** | | |

> Note: The robber’s position is implicitly encoded by tile flags; no extra global field is added.

---

## 2️⃣ Self Information (Player 1)

Private information and stats about the observing agent.  
All scalar counts are normalized to `[0, 1]` based on maximum theoretical or typical values.

| Feature | Count | Description | Normalization |
|----------|--------|-------------|----------------|
| Resource counts | 5 | Amount of each resource | ÷ 19 |
| Development cards (by type) | 5 | Owned but unplayed | ÷ 5 |
| Victory points | 1 | Total victory points | ÷ 10 |
| Longest road flag | 1 | Boolean (0/1) | — |
| Largest army flag | 1 | Boolean (0/1) | — |
| Built structures | 3 | # of roads, settlements, cities built | ÷ [15, 5, 4] |
| Played knights | 1 | Count of played knight cards | ÷ 14 |
| **Total (Self)** | **17** | | |

---

## 3️⃣ Others (Public Opponent Information)

Limited to publicly visible or inferable information for each of the 3 other players.

| Feature | Count per opponent | Description |
|----------|--------------------|-------------|
| Roads built | 1 | Count of placed roads |
| Settlements built | 1 | Count of settlements |
| Cities built | 1 | Count of cities |
| Development cards (total) | 1 | Number of unplayed dev cards |
| Victory points | 1 | Public + estimated total |
| Longest road flag | 1 | Boolean |
| Largest army flag | 1 | Boolean |
| Knights played | 1 | Count of played knight cards |
| **Total per opponent** | **8** | |
| **Total (3 opponents)** | **24** | |

---

## 🧮 Final Observation Shape Summary

| Section | Length | Description |
|----------|---------|-------------|
| Global board | 935 | All public board information |
| Self info | 17 | Private normalized stats of observing player |
| Others info | 24 | Limited stats of 3 opponents |
| **Total observation size** | **976** | Final flattened observation vector length |

---

## ⚙️ Design Principles

- **Fixed-length**: Stable vector size regardless of game phase.
- **Agent-relative rotation**: Each agent sees itself as Player 1.
- **No redundant encoding**: Robber, dice, or duplicate flags avoided.
- **Consistent normalization**: Only continuous or count-based features normalized.
- **Graph-consistent**: Spatial adjacency (tile–node–edge) can be added via constant matrices later.

---

## 📊 Normalization Ranges

| Feature Type | Range | Example |
|---------------|--------|----------|
| Resource count | [0, 19] | Divide by 19 |
| Victory points | [0, 10] | Divide by 10 |
| Roads | [0, 15] | Divide by 15 |
| Settlements | [0, 5] | Divide by 5 |
| Cities | [0, 4] | Divide by 4 |
| Knights | [0, 14] | Divide by 14 |
| Boolean flags | {0, 1} | Unchanged |
| One-hot encodings | {0, 1} | Unchanged |

---

## 🧠 Implementation Notes

- The observation vector is assembled as:
  ```python
  obs = np.concatenate([
      global_board_features,
      self_features,
      others_features
  ])
