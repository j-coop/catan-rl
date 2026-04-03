# **Catan MARL — Reward System Architecture**

This document summarizes the final reward shaping design for the multi-agent Catan environment.  
The system uses **potential-based reward shaping**, where the reward is derived from *changes in a potential function* between states. This provides dense learning signals without altering the underlying optimal policy.

---

## **1. Why Potential-Based Rewards?**

Traditional sparse rewards (e.g., +1 for building, +10 for winning) are insufficient for Catan:

- Many important actions have **delayed consequences**.
- Agents must plan across **multiple turns**, balancing tempo, risk, and probabilistic gains.
- Standard “shaping rewards” can unintentionally change the optimal policy.

**Potential-based shaping** avoids these issues by defining a scalar function Φ(s) describing how good a state is.

**Reward(agent, action) = Φ(s′) − Φ(s)**

### Benefits:
- **Dense, informative rewards** without reward hacking.
- The optimal policy is preserved theoretically.
- Captures complex strategic progress (economy, safety, ports, dev cards).
- Encourages long-term planning.
- Learns placement and engine-building strategy more effectively.

---

## **2. Potential Function Overview**

The potential is computed per-player as:

Potential =
- VictoryPoints
- ExpectedProduction
- ResourceHoldings
- DevCards
- Ports
- Roads
- Risk


Each component reflects a major dimension of strategic advantage.

### Component Overview

| Component | Meaning | Why It Matters |
|----------|----------|----------------|
| **Victory Points** | Core progress | True win objective |
| **Expected Production** | Long-term resource flow & diversity | Main determinant of future tempo |
| **Resource Holdings** | Current ability to build or trade | Encourages using strong hands |
| **Development Cards** | Knight tempo, special actions | Supports diverse strategic options |
| **Ports** | Trading potential | Synergy with resource profile |
| **Roads** | Expansion ability | Keeps expansion from being punished |
| **Risk** | Robber blocking, card-loss risk | Encourages safe states |

---

## **3. Detailed Component Table**

### **3.1 Victory Points**
| Metric | Description |
|--------|-------------|
| VP / 10 | Scaled to [0,1] then heavily weighted |

VP is the strongest signal and must stay dominant.

---

### **3.2 Expected Production Component**
**Formula:**  
`0.6 * quantity + 0.4 * entropy`

| Subcomponent | Description | Notes |
|--------------|-------------|-------|
| **Quantity** | Expected resource generation weighted by dice probabilities and resource bias | Cities count double |
| **Resource Bias** | wood/brick/wheat slightly ↑; sheep/ore slightly ↓ | Matches early-game engine needs |
| **Entropy** | Production diversity measure | Encourages a resilient economy |

---

### **3.3 Resource Holdings Component**
| Count | Reward | Meaning |
|-------|--------|---------|
| 1st resource of type | +0.3 | Foundational |
| 2nd resource | +0.2 | Still strong |
| 3rd+ | +0.1 each | Diminishing returns |

Encourages useful, diverse hands without scaling too strongly.

---

### **3.4 Development Card Value**

| Card Type | Value |
|-----------|--------|
| Unplayed Knight | +0.4 |
| Played Knights | +0.3 (army progress) |
| Road Building | +0.5 |
| Monopoly | +0.6 |
| Year of Plenty | +0.4 |

(Dev-card VPs are counted in VP component.)

---

### **3.5 Port Value Component**

| Port Type | Contribution |
|-----------|--------------|
| Specific resource port | Weighted by production of that resource (6-token equivalence) |
| Generic 3:1 | +0.3 |

Rewards synergy between economy and trading infrastructure.

---

### **3.6 Roads Component**
| Rule | Meaning |
|------|----------|
| +2.5 + 3.0×Quality | Bonus for opening new settlement spots (scaled by adjacent production yield) |
| -0.75 Penalty | Penalty for true dead-ends (leads to nodes blocked by opponents) |
| +2.0 Bonus | For "Connecting" segments (longest road jump $\ge 3$) |
| +1.0 Base | Baseline reward for general road placements |

---

### **3.7 Risk Penalty**
Weighted combination of:

| Risk | Description |
|------|-------------|
| **Card-loss risk** | Cards above 7 incur increasing penalty |
| **Robber blocking** | Settlement/city adjacent to blocked tile incurs penalty |

Encourages safe hand-size and robber management.

---

## **4. Final Potential Formula**

Potential =
- 5.0 * VictoryPoints
- 1.0 * ExpectedProduction
- 0.3 * ResourceHoldings
- 0.2 * DevCards
- 0.2 * Ports
- 0.2 * Roads
- (-0.2 * Risk)

---

## **5. Important Remarks**

### **1. Advantages Over Fixed Rewards**
- No reward manipulation.
- No bias toward short-term actions.
- Better long-horizon decision making.
- Replaces dozens of noisy heuristic rewards with a single stable signal.

### **2. Why No Direct Rewards for Some Actions**
- **No reward for “build settlement” or “build city”**  
  → Their impact enters through VP, production, maps/ports, resource flow.

- **No large reward for Largest Army / Longest Road**  
  → They can be lost; instead, victory points, production, roads, and knights played capture their strategic value.

- **No direct reward for getting resources**  
  → Production potential, risk management, and resource holdings account for their usefulness.

### **3. Ultimate objective**
- **Actually winning the game provides a fixed terminal reward of +7.0.**
- **Losing the game providing a fixed terminal reward of -7.0.**
- **Reward Signal:** The environment provides a combination of Potential-Based Reward Shaping (PBRS) and specific strategic heuristics to guide training. Terminal win/loss signals remain the dominant objective.

---

