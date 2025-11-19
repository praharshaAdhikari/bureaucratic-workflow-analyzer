"""
Main execution script for workflow simulation.

This script runs the workflow simulation by orchestrating the different
components: reading logs, setting up workers, running the simulation,
and generating reports.
"""

import simpy

from digital_twin.workflow import Workflow


def main():
    """Main execution function."""
    # Create simulation environment
    env = simpy.Environment()

    # Create workflow instance
    workflow = Workflow("Main Process", env)

    # Load and process logs
    logs_df = workflow.read_logs()
    if logs_df is not None:
        # Setup ideal durations and workers
        workflow.read_ideal_durations()
        workflow.define_workers()

        # Start simulation
        workflow.start(speed=100)

        # Run the simulation
        env.run()

        # Generate final report
        workflow.generate_report()


if __name__ == "__main__":
    main()
