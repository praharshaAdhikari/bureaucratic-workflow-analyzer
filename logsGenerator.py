import csv
import random

# -----------------------------
# CONFIGURATION
# -----------------------------
NUM_CASES = 100            # Total number of cases
TASKS_PER_CASE = (3, 6)    # Min and max tasks per case
WORKERS = ["WorkerA", "WorkerB", "WorkerC"]
TASK_TYPES = ["Review", "Approval", "Processing", "Verification", "Inspection"]
TIME_STEP = (2, 6)         # Duration of each task in minutes
MAX_START_GAP = 3           # Max random gap between new cases
OUTPUT_FILE = "logs.csv"

# -----------------------------
# GENERATE SYNTHETIC LOGS
# -----------------------------
logs = []
current_time = 0

for case_id in range(1, NUM_CASES + 1):
    num_tasks = random.randint(*TASKS_PER_CASE)
    
    # Random start time with slight overlap between cases
    task_start = current_time + random.randint(0, MAX_START_GAP)
    
    for _ in range(num_tasks):
        task = random.choice(TASK_TYPES)
        worker = random.choice(WORKERS)
        duration = random.randint(*TIME_STEP)
        task_end = task_start + duration
        
        logs.append([case_id, task, worker, task_start, task_end])
        
        # Next task in the same case starts right after the previous
        task_start = task_end
    
    # Allow next case to start slightly overlapping previous case
    current_time += random.randint(1, 4)

# -----------------------------
# SAVE TO CSV
# -----------------------------
with open(OUTPUT_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["CaseID", "Task", "Worker", "StartTime", "EndTime"])
    writer.writerows(logs)

print(f"Synthetic workflow logs generated: {OUTPUT_FILE}")
print(f"Total cases: {NUM_CASES}, Total tasks: {len(logs)}")
