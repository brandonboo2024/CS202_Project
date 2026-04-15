import sys
import time
import heapq
import random
import csv
import os

# ==========================================
# 1. PARSER MODULE
# ==========================================
def parse_psplib(file_path):
    """
    Parses a pure numerical .SCH file block by block.
    Dynamically supports both RCPSP/max (with mode/lags) and Standard RCPSP formats.
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    # 1. Parse Metadata (First line)
    meta_data = lines[0].split()
    n = int(meta_data[0]) 
    K = int(meta_data[1])
    
    # DETECT FORMAT: If the first line has > 2 items (like "10 5 0 0"), it has a Mode column.
    has_mode = len(meta_data) > 2
    
    durations = [0] * (n + 2)
    resource_reqs = [[0] * K for _ in range(n + 2)]
    successors = [[] for _ in range(n + 2)]
    predecessors_count = [0] * (n + 2)
    
    # 2. Parse Precedence Block (Next n + 2 lines)
    for i in range(1, n + 3):
        parts = lines[i].split()
        task_id = int(parts[0])
        
        # Adjust target columns based on file format
        if has_mode:
            num_successors = int(parts[2])
            succ_start_idx = 3
        else:
            num_successors = int(parts[1])
            succ_start_idx = 2
            
        succ_list = []
        for j in range(num_successors):
            succ_id = int(parts[succ_start_idx + j])
            
            if succ_id > task_id:
                succ_list.append(succ_id)
                predecessors_count[succ_id] += 1 
                
        successors[task_id] = succ_list

    for i in range(1, n + 1):
        if len(successors[i]) == 0:
            successors[i].append(n + 1)
            predecessors_count[n + 1] += 1
        
    # 3. Parse Duration & Resource Block (Next n + 2 lines)
    offset = n + 3
    for i in range(offset, offset + n + 2):
        parts = lines[i].split()
        task_id = int(parts[0])
        
        # Adjust target columns based on file format
        if has_mode:
            durations[task_id] = int(parts[2])
            req_start_idx = 3
        else:
            durations[task_id] = int(parts[1])
            req_start_idx = 2
            
        for k in range(K):
             resource_reqs[task_id][k] = int(parts[req_start_idx + k])
             
    # 4. Parse Resource Capacities (Last line)
    capacity_line = lines[offset + n + 2].split()
    capacities = [int(x) for x in capacity_line]
    
    return n, K, durations, resource_reqs, successors, predecessors_count, capacities

# ==========================================
# 2. THE PARALLEL SGS ENGINE
# ==========================================
def parallel_sgs_grasp(n, K, durations, resource_reqs, successors, capacities, predecessors_count, heuristic_scores, alpha=3):
    """
    Generates a single, guaranteed-valid schedule.
    alpha: The Restricted Candidate List (RCL) size for GRASP randomness.
    
    """
    # Create local copies of mutable state arrays
    in_degree = predecessors_count[:]
    available_resources = capacities[:]
    
    # Task 0 is the dummy start. If it has 0 incoming edges, it is eligible.
    eligible = [i for i in range(n + 2) if in_degree[i] == 0]
    
    start_times = [-1] * (n + 2)
    active_tasks = [] # Min-heap tracking completions: (finish_time, task_id)
    
    current_time = 0
    scheduled_count = 0
    
    while scheduled_count < n + 2:
        scheduled_this_step = False
        
        # 1. Find which eligible tasks fit within current resources
        schedulable = []
        for task in eligible:
            fits = True
            for k in range(K):
                if resource_reqs[task][k] > available_resources[k]:
                    fits = False
                    break
            if fits:
                schedulable.append(task)
                
        # 2. Schedule tasks if possible
        if schedulable:
            # Sort by our pre-calculated heuristic (e.g., descending order)
            schedulable.sort(key=lambda x: heuristic_scores[x], reverse=True)
            
            # GRASP: Pick randomly from the top 'alpha' candidates
            limit = min(alpha, len(schedulable))
            chosen_idx = random.randint(0, limit - 1)
            task_to_schedule = schedulable[chosen_idx]
            
            # Record start time and push to active timeline
            start_times[task_to_schedule] = current_time
            finish_time = current_time + durations[task_to_schedule]
            heapq.heappush(active_tasks, (finish_time, task_to_schedule))
            
            # Deduct resources
            for k in range(K):
                available_resources[k] -= resource_reqs[task_to_schedule][k]
                
            eligible.remove(task_to_schedule)
            scheduled_count += 1
            scheduled_this_step = True
            
        # 3. If we couldn't schedule anything, fast-forward time
        if not scheduled_this_step and active_tasks:
            # Jump to the time of the next finishing task
            next_time, finished_task = heapq.heappop(active_tasks)
            current_time = next_time
            
            # Release its resources
            for k in range(K):
                available_resources[k] += resource_reqs[finished_task][k]
                
            # Unlock successors (mimicking topological sort step)
            for succ in successors[finished_task]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    eligible.append(succ)
                    
            # Check if any other tasks finished at this exact same millisecond
            while active_tasks and active_tasks[0][0] == current_time:
                _, finished_task = heapq.heappop(active_tasks)
                for k in range(K):
                    available_resources[k] += resource_reqs[finished_task][k]
                for succ in successors[finished_task]:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        eligible.append(succ)
                        
        elif not scheduled_this_step and not active_tasks:
            return float('inf'), [] # Return infinite makespan so GRASP ignores it

    # The makespan is the start time of the dummy end node (n + 1)
    return start_times[n + 1], start_times

def validate_schedule(start_times, n, K, predecessors_count, durations, resource_reqs, successors, capacities):
    """
    Checks a generated schedule for precedence and resource violations.
    Modified to be SILENT on success so it doesn't flood the console during batch processing.
    """
    in_degree = predecessors_count[:]
    topo_order = []
    queue = [i for i in range(n + 2) if in_degree[i] == 0]
    
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for succ in successors[node]:
            if succ > node: 
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
                    
    if len(topo_order) != n + 2:
        return False, "CYCLE DETECTED"

    for j in topo_order:
        for succ in successors[j]:
            if succ > j:
                finish_j = start_times[j] + durations[j]
                if start_times[succ] < finish_j:
                    return False, f"PRECEDENCE VIOLATION: Task {succ} starts before {j} finishes."
    
    makespan = start_times[n + 1]
    
    for t in range(makespan):
        current_usage = [0] * K
        for i in range(1, n + 1):
            if start_times[i] <= t < (start_times[i] + durations[i]):
                for k in range(K):
                    current_usage[k] += resource_reqs[i][k]
                    
        for k in range(K):
            if current_usage[k] > capacities[k]:
                return False, f"RESOURCE VIOLATION at t={t} for resource {k}."

    return True, "VALID"

# ==========================================
# 3. SINGLE INSTANCE SOLVER
# ==========================================
def solve_instance(file_path, time_limit=0.2):
    """
    Solves a single instance and returns the best makespan and start times.
    """
    start_timer = time.perf_counter()
    n, K, durations, resource_reqs, successors, predecessors_count, capacities = parse_psplib(file_path)
    
    heuristic_scores = [0] * (n + 2)
    for i in range(n + 2):
        heuristic_scores[i] = durations[i] + len(successors[i])
        
    best_makespan = float('inf')
    best_start_times = []
    
    iterations = 0
    while time.perf_counter() - start_timer < time_limit:
        current_alpha = 1 if iterations == 0 else random.randint(2, 5)
        
        makespan, s_times = parallel_sgs_grasp(
            n, K, durations, resource_reqs, successors, capacities, 
            predecessors_count, heuristic_scores, alpha=current_alpha
        )
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_start_times = s_times
            
        iterations += 1

    # Validate before returning
    if best_start_times:
        is_valid, msg = validate_schedule(best_start_times, n, K, predecessors_count, durations, resource_reqs, successors, capacities)
        if not is_valid:
            return "INVALID", msg
            
        # Return makespan and the start times for tasks 1 through n
        return best_makespan, best_start_times[1:n+1] 
    else:
        return "FAILED", "No schedule found"

# ==========================================
# 4. BATCH PROCESSOR
# ==========================================
def run_batch():
    # The folders you want to scan
    folders_to_scan = ['sm_j10', 'sm_j20']
    
    # Time limit PER INSTANCE (Set to 2.0 seconds for quick testing)
    TIME_BUDGET_PER_FILE = 0.2
    
    output_filename = "batch_results.csv"
    
    print(f"Starting batch process. Output will be saved to {output_filename}...")
    
    # Open a CSV file to write our table
    with open(output_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write the header row
        writer.writerow(["Directory", "Filename", "Makespan", "Start_Times (1 to n)", "Status"])
        
        for folder in folders_to_scan:
            # Check if the folder actually exists on your computer
            if not os.path.exists(folder):
                print(f"Warning: Folder '{folder}' not found. Skipping.")
                continue
                
            # Iterate through all files in the directory
            for filename in os.listdir(folder):
                # Look for .SCH, .sm, or .rcp files
                if filename.upper().endswith('.SCH') or filename.upper().endswith('.SM'):
                    file_path = os.path.join(folder, filename)
                    
                    print(f"Processing {folder}/{filename}...", end="", flush=True)
                    
                    # Run the algorithm!
                    makespan, result_data = solve_instance(file_path, time_limit=TIME_BUDGET_PER_FILE)
                    
                    # Format output for the CSV
                    if isinstance(makespan, int):
                        # Success
                        status = "Valid"
                        start_times_str = str(result_data)
                        print(f" Done! Makespan: {makespan}")
                    else:
                        # Failed or Invalid
                        status = result_data # the error message
                        start_times_str = "[]"
                        print(f" Error: {status}")
                        
                    # Write the row to the table
                    writer.writerow([folder, filename, makespan, start_times_str, status])

    print(f"\nBatch processing complete! Check {output_filename} for your table.")

if __name__ == "__main__":
    # If the user provides a file in the console (e.g. python solver.py PSP1.SCH), just solve that one file.
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Solving single instance: {file_path}")
        makespan, start_times = solve_instance(file_path, time_limit=29.5)
        if isinstance(makespan, int):
             for st in start_times:
                 print(st)
        else:
             print(f"Failed: {start_times}")
             
    # If the user just runs `python solver.py` with no arguments, run the batch script!
    else:
        run_batch()