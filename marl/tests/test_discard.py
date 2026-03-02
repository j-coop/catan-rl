import sys
sys.path.append('.')

from marl.model.DiscardManager import DiscardManager
from params.catan_constants import BUILD_COSTS

def main():
    print("Testing DiscardManager Heuristic")
    tests = [
        {
            "name": "Hoarder of Sheep (early game)",
            "resources": {"wood": 1, "brick": 1, "sheep": 6, "wheat": 0, "ore": 0}, 
            "production_probs": {"wood": 0.1, "brick": 0.1, "sheep": 0.3, "wheat": 0.1, "ore": 0.0},
            "victory_points": 2,
        },
        {
            "name": "Needs Ore/Wheat Late Game",
            "resources": {"wood": 4, "brick": 0, "sheep": 0, "wheat": 2, "ore": 2}, 
            "production_probs": {"wood": 0.2, "brick": 0.0, "sheep": 0.0, "wheat": 0.0, "ore": 0.0},
            "victory_points": 8,
        },
        {
            "name": "One wheat away from city",
            "resources": {"wood": 4, "brick": 0, "sheep": 0, "wheat": 1, "ore": 3},
            "production_probs": {"wood": 0.1, "brick": 0.1, "sheep": 0.1, "wheat": 0.1, "ore": 0.1},
            "victory_points": 5,
        },
        {
            "name": "What now?",
            "resources": {"wood": 2, "brick": 1, "sheep": 1, "wheat": 1, "ore": 3},
            "production_probs": {"wood": 0.1, "brick": 0.1, "sheep": 0.1, "wheat": 0.1, "ore": 0.1},
            "victory_points": 5,
        }
    ]

    for t in tests:
        print(f"\n--- Scenario: {t['name']} ---")
        print(f"VP: {t['victory_points']} | Probs: {t['production_probs']}")
        print(f"Hand: {t['resources']}")
        discarded = DiscardManager.get_cards_to_discard(
            t["resources"], 
            t["production_probs"], 
            t["victory_points"], 
            BUILD_COSTS
        )
        print(f"Discarded ({len(discarded)} cards): {discarded}")
        
        after = dict(t["resources"])
        for d in discarded:
            after[d] -= 1
        print(f"Kept: {after}")

if __name__ == "__main__":
    main()
