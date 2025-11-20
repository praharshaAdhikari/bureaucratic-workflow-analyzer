"""
Main execution script for workflow simulation.

This script runs the workflow simulation by orchestrating the different
components: reading logs, setting up workers, running the simulation,
and generating reports.
"""

import os

import simpy

from digital_twin.sim_env import SimEnv


def main():
    """Main execution function."""
    # Create simulation environment
    env = simpy.Environment()

    # Define worker resources (example)
    workers = []  # Replace with actual worker names
    twin_params = {"workers": workers}

    # Create SimEnv instance, passing in the env
    sim_env = SimEnv(twin_params=twin_params)
    sim_env.env = env
    sim_env.reset()

    # Run the simulation
    env.run(until=24 * 60)
    # Generate final report
    # workflow.generate_report() # there is no workflow, so comment this out.


if __name__ == "__main__":
    main()
