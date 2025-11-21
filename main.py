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
    # Create simulation environment
    env = simpy.Environment()

    workflow = Workflow("Workflow", env)
    workflow.read_logs()
    workers = list(workflow.logs["Worker"].unique())

    twin_params = {"workers": workers}

    # Create SimEnv instance, passing in the env
    sim_env = SimEnv(twin_params=twin_params)
    sim_env.env = env
    sim_env.reset()

    # Run the simulation
    # env.run(until=24 * 60)  # example: run for 24 hours

    # Generate final report
    # workflow.generate_report()

    # Train the RL agent:
    train(sim_env)

    # Evaluate RL agent
    # evaluate(sim_env)


if __name__ == "__main__":
    main()
