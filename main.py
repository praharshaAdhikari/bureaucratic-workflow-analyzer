"""
Main execution script for workflow simulation.
"""

import os

import simpy

from digital_twin.sim_env import SimEnv
from digital_twin.workflow import Workflow
from rl.evaluate_rl import evaluate
from rl.train_rl import train


def main():
    """Main execution function."""
    train()
    # evaluate()


if __name__ == "__main__":
    main()
