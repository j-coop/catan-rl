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

Encodes the full visible structure of the game board: tiles, roads, nodes (including node-level port info), and robber.

| Subcomponent | Count | Feature breakdown | Description |
|---------------|--------|-------------------|--------------|
| **Tiles (19)** | 19 × 16 = 304 | - 6 one-hot resource types (wood, brick, wheat, sheep, ore, desert) <br> - 1 normalized number token (`2–12 → /12`) <br> - 1 robber flag <br> - **4 productivity metrics** (self + 3 opponents) <br> - **4 building presence flags** (self + 3 opponents) | Resource production, spatial productivity explicitly mapped per player, and robber position |
| **Roads (72)** | 72 × 5 = 360 | - 5 one-hot ownership (4 players + empty) | Tracks who owns each road |
| **Nodes (54)** | 54 × 13 = 702 | - 5 one-hot ownership (4 players + empty) <br> - 2-hot building type (settlement, city) <br> - 6 one-hot port type (wood, brick, wheat, sheep, ore, generic; all-zero = no port) | Building placement, ownership, and which nodes grant port access |
| **Total (Global)** | **304 + 360 + 702 = 1,366** | | |

> ⚙️ **Ports are encoded per node** (as a 6-dim one-hot inside each node feature). Ownership/use of ports is kept in player-level features (see Self / Others sections).

> 💡 **Potential extension:** Track how many resource cards each node’s settlement/city has generated so far (per resource type). This could provide valuable reinforcement signal for learning spatial strategy but is currently omitted to control observation size.

> Note: The robber’s position is encoded via tile flags; no separate global robber field is added.

---

## 2️⃣ Self Information (Player 1)

Private information and stats about the observing agent.  
All scalar counts are normalized to `[0, 1]` based on maximum theoretical or typical values.

| Feature | Count | Description | Normalization |
|----------|--------|-------------|----------------|
| Total resource count | 1 | Sum of all resource cards in hand | ÷ 20 |
| Resource counts | 5 | Amount of each resource (wood, brick, sheep, wheat, ore) | ÷ 19 |
| Resource production probabilities | 5 | Aggregated probability of producing each resource per turn (Settlement=1x, City=2x) | Cap at 1.0 |
| Development cards (by type) | 5 | Owned but unplayed | ÷ 5 |
| Victory points | 1 | Total victory points | ÷ 10 |
| Longest road flag | 1 | Boolean (0/1) | — |
| Largest army flag | 1 | Boolean (0/1) | — |
| Built structures | 3 | # of roads, settlements, cities built | ÷ [15, 5, 4] |
| Played knights | 1 | Count of played knight cards | ÷ 14 |
| Ports owned | 6 | One-hot for accessible port types (aggregated across owned nodes) | — |
| **Total (Self)** | **29** | | |

---

## 3️⃣ Others (Public Opponent Information)

Limited to publicly visible or inferable information for each of the 3 other players.

| Feature | Count per opponent | Description | Normalization |
|----------|--------------------|--------------|----------------|
| Total resource count | 1                  | Sum of all resource cards in hand | ÷ 20 |
| Roads built | 1                  | Count of placed roads | ÷ 15 |
| Settlements built | 1                  | Count of settlements | ÷ 5 |
| Cities built | 1                  | Count of cities | ÷ 4 |
| Resource production probabilities | 5 | Aggregated probability of producing each resource per turn | Cap at 1.0 |
| Development cards (total) | 1                  | Number of unplayed dev cards | ÷ 10 |
| Victory points | 1                  | Public + estimated total | ÷ 10 |
| Longest road flag | 1                  | Boolean (0/1) | — |
| Largest army flag | 1                  | Boolean (0/1) | — |
| Knights played | 1                  | Played knight cards | ÷ 14 |
| Ports owned | 6                  | One-hot for accessible port types | — |
| **Total per opponent** | **20**             | | |
| **Total (3 opponents)** | **60**             | | |

---

## 🧮 Final Observation Shape Summary

| Section | Length    | Description |
|----------|-----------|-------------|
| Global board | 1,366     | Tiles (with per-player productivity), roads, nodes |
| Self info | 29        | Private stats (Total Count, Resources, Production) |
| Others info | 60        | Stats of 3 opponents (Hand size, Production, Built) |
| **Total observation size** | **1,455** | Final flattened observation vector length |

---

## ⚙️ Design Principles

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
