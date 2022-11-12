from society.agent import Agent
from society.simulation import Simulation
from society.strategies.gameplay.constant import AllC
from society.strategies.gameplay.random import RandomGameplayStrategy
from society.strategies.selection.greedy import GreedySelectionStrategy
from society.strategies.selection.random import RandomSelectionStrategy


def main():
    sim = Simulation(
        [
            Agent(GreedySelectionStrategy(), AllC()),
            Agent(RandomSelectionStrategy(), AllC()),
            Agent(GreedySelectionStrategy(), RandomGameplayStrategy()),
            Agent(RandomSelectionStrategy(), RandomGameplayStrategy()),
        ]
    )

    print(sim.returns)

    sim.play_round()

    print(sim.returns)


if __name__ == "__main__":
    main()
