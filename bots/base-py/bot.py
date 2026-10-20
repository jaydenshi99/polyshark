"""base-py — the smallest complete Python bot. Copy this folder to start a new one."""
import random

from polyshark_bot import Bot, Decision, run


class BaseBot(Bot):
    name = "base-py"
    version = "1"

    def init(self, hello):
        self.player = hello["player"]
        self.rng = random.Random(hello["seed"])  # the only randomness source to use

    def act(self, obs):
        return Decision(self.choose(obs))

    def analyse(self, obs, budget):
        """Position Analyser hook: the current evaluation, no move scores yet."""
        return {"eval": self.evaluate(obs), "moves": []}

    def choose(self, obs):
        """Strategy goes here. For now: end the turn, or take the first option
        when a city upgrade must be chosen (EndTurn is not offered until then)."""
        end = obs.find("EndTurn")
        return end if end is not None else 0

    def evaluate(self, obs):
        """Evaluation of obs for this bot, in [-1, 1]."""
        return 0.0


if __name__ == "__main__":
    run(BaseBot())
