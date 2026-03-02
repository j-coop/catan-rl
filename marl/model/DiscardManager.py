from typing import Dict, List
from collections import Counter
import random

class DiscardManager:
    @staticmethod
    def get_cards_to_discard(
        resources: Dict[str, int], 
        production_probs: Dict[str, float],
        victory_points: int,
        build_costs: Dict[str, Dict[str, int]]
    ) -> List[str]:
        total_cards = sum(resources.values())
        if total_cards <= 7:
            return []
            
        to_discard_count = total_cards // 2
        
        # 1. Base Utility
        base_utility = {
            "wheat": 5,
            "wood": 4,
            "brick": 4,
            "ore": 3,
            "sheep": 1
        }
        
        # 2. Phase-based Priority
        phase_bonus = {"wood": 0, "brick": 0, "sheep": 0, "wheat": 0, "ore": 0}
        if victory_points <= 4:
            phase_bonus["wood"] += 3
            phase_bonus["brick"] += 3
        elif victory_points >= 6:
            phase_bonus["ore"] += 3
            phase_bonus["wheat"] += 3
            
        # 3. Scarcity / Production Capability
        scarcity_bonus = {}
        for res, prob in production_probs.items():
            if prob == 0:
                scarcity_bonus[res] = 5
            elif prob < (3/36):
                scarcity_bonus[res] = 3
            else:
                scarcity_bonus[res] = 0
                
        # 4. Marginal Cost Protection (Settlements and Cities only)
        # Find closest goal
        goals = ["settlement", "city"]
        closest_goal_missing = float('inf')
        closest_goal = None
        
        for goal in goals:
            cost = build_costs[goal]
            missing = 0
            for res, req in cost.items():
                if resources.get(res, 0) < req:
                    missing += (req - resources.get(res, 0))
            if missing < closest_goal_missing:
                closest_goal_missing = missing
                closest_goal = goal
                
        protection_bonus = {"wood": 0, "brick": 0, "sheep": 0, "wheat": 0, "ore": 0}
        if closest_goal:
            cost = build_costs[closest_goal]
            for res, req in cost.items():
                if resources.get(res, 0) > 0:
                    protection_bonus[res] = 20
        
        # Expand hand to individual cards
        hand = []
        for res, count in resources.items():
            hand.extend([res] * count)
            
        # We will score each card. But wait, surplus penalty depends on how many we keep.
        # Instead, we can score them incrementally or just apply surplus penalty per idx.
        # For a given resource, the 1st card kept has 0 penalty, 2nd has 0, 3rd has -2, 4th has -4, etc.
        # So we can calculate the score for the i-th instance of the resource.
        scored_cards = []
        res_counts = {"wood": 0, "brick": 0, "sheep": 0, "wheat": 0, "ore": 0}
        
        for res in hand:
            idx = res_counts[res]
            res_counts[res] += 1
            
            # surplus penalty
            surplus_penalty = 0
            if idx >= 2:
                surplus_penalty = (idx - 1) * -2
                
            # For protection bonus, only the cards that are ACTUALLY needed for the goal get the massive bonus
            # Wait, the instruction said "if its inclusion doesn't exceed the required amount for that goal".
            # So if we need 1 wheat, only the first wheat gets +20.
            current_protection = 0
            if closest_goal:
                req = build_costs[closest_goal].get(res, 0)
                if idx < req:
                    current_protection = 20
            
            score = (
                base_utility.get(res, 0) +
                phase_bonus.get(res, 0) +
                scarcity_bonus.get(res, 0) +
                current_protection +
                surplus_penalty
            )
            
            # Add some small random noise to break ties dynamically
            score += random.uniform(0, 0.1)
            
            scored_cards.append((score, res))
            
        # Sort descending by score
        scored_cards.sort(key=lambda x: x[0], reverse=True)
        
        keep_count = total_cards - to_discard_count
        kept = scored_cards[:keep_count]
        discarded = scored_cards[keep_count:]
        
        return [res for score, res in discarded]
