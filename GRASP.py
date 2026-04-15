import sys
import time
import heapq
import random

# ==========================================
# 1. PARSER MODULE
# ==========================================
def parse_psplib(file_path):
    """
    Parses a pure numerical .SCH file block by block.
    """
    with open(file_path, 'r') as f:
        # Read all lines, strip whitespace, and ignore completely empty lines
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    # 1. Parse Metadata (First line)
    # Format: n (jobs), K (resources), [flag], [flag]
    meta_data = lines[0].split()
    n = int(meta_data[0]) 
    K = int(meta_data[1])
    
    # Initialize arrays (n + 2 to account for dummy start 0 and dummy end n+1)
    durations = [0] * (n + 2)
    resource_reqs = [[0] * K for _ in range(n + 2)]
    successors = [[] for _ in range(n + 2)]
    predecessors_count = [0] * (n + 2)
    
    # 2. Parse Precedence Block (Next n + 2 lines)
    # Format: TaskID, mode, num_successors, succ1, succ2... [lag1], [lag2]...
    # We loop from line index 1 to (n + 2)
    for i in range(1, n + 3):
        parts = lines[i].split()
        task_id = int(parts[0])
        num_successors = int(parts[2])
        
        succ_list = []
        for j in range(num_successors):
            succ_id = int(parts[3 + j])
            
            # THE FIX: Only accept forward edges to maintain a strict DAG
            if succ_id > task_id:
                succ_list.append(succ_id)
                predecessors_count[succ_id] += 1 
                
        successors[task_id] = succ_list
        
    # 3. Parse Duration & Resource Block (Next n + 2 lines)
    # Format: TaskID, mode, duration, req1, req2... reqK
    offset = n + 3
    for i in range(offset, offset + n + 2):
        parts = lines[i].split()
        task_id = int(parts[0])
        durations[task_id] = int(parts[2])
        
        # Extract resource requirements for this task
        for k in range(K):
             resource_reqs[task_id][k] = int(parts[3 + k])
             
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

# ==========================================
# 3. MAIN EXECUTION & TIME MANAGEMENT
# ==========================================
def solve():
    # Perf_counter is the highest resolution clock in Python
    time_limit = 29.5 
    start_timer = time.perf_counter()
    
    if len(sys.argv) < 2:
        print("Usage: python solver.py <instance_file.SCH>")
        return

    file_path = sys.argv[1]
    
    # 1. Parse File
    n, K, durations, resource_reqs, successors, predecessors_count, capacities = parse_psplib(file_path)
    
    # 2. Precompute Static Heuristic
    # Simple example: prioritize tasks that take a long time and unlock many successors
    heuristic_scores = [0] * (n + 2)
    for i in range(n + 2):
        heuristic_scores[i] = durations[i] + len(successors[i])
        
    best_makespan = float('inf')
    best_start_times = []
    
    # 3. GRASP Loop (The Anytime Algorithm)
    iterations = 0
    while time.perf_counter() - start_timer < time_limit:
        
        # Pass 1: Force purely greedy (alpha=1) to guarantee at least one safe baseline
        current_alpha = 1 if iterations == 0 else random.randint(2, 5)
        
        makespan, s_times = parallel_sgs_grasp(
            n, K, durations, resource_reqs, successors, capacities, 
            predecessors_count, heuristic_scores, alpha=current_alpha
        )
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_start_times = s_times
            
        iterations += 1

    # 4. Strict Formatting Output
    # The guidelines state: "Print start times for activities 1 through n to stdout, one integer per line"
    # Run the validation check
    is_valid = validate_schedule(
        best_start_times, n, K, predecessors_count, durations, resource_reqs, 
        successors, capacities
    )
    
    # Only print the stdout if it actually passed
    if is_valid:
        for i in range(1, n + 1):
            print(best_start_times[i])
    else:
        print("Algorithm failed to find a valid schedule.")
        
        
        
def validate_schedule(start_times, n, K, predecessors_count, durations, resource_reqs, successors, capacities):
    """
    Checks a generated schedule for precedence and resource violations.
    Returns True if valid, False otherwise.
    """
    print("\n--- Running Schedule Validation ---")
    
    # ---------------------------------------------------------
    # 1. PRECEDENCE CHECK
    # ---------------------------------------------------------
    in_degree = predecessors_count[:]
    topo_order = []
    queue = [i for i in range(n + 2) if in_degree[i] == 0]
    
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for succ in successors[node]:
            # Remember: We only care about forward edges
            if succ > node: 
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
                    
    if len(topo_order) != n + 2:
        print("❌ CRITICAL ERROR: The graph contains a cycle!")
        return False

    # Now sweep through the start times in perfect topological order
    for j in topo_order:
        for succ in successors[j]:
            if succ > j:
                finish_j = start_times[j] + durations[j]
                if start_times[succ] < finish_j:
                    print(f"❌ PRECEDENCE VIOLATION detected during Topo Sweep:")
                    print(f"   Task {succ} starts at {start_times[succ]} before Task {j} finishes at {finish_j}.")
                    return False
    
    print("✅ Precedence Constraints: Passed")

    # ---------------------------------------------------------
    # 2. RESOURCE CHECK (Time-Step Sweep)
    # ---------------------------------------------------------
    makespan = start_times[n + 1]
    
    # We check every single time unit from t=0 up to the makespan
    for t in range(makespan):
        current_usage = [0] * K
        
        # Look at every real task (1 to n)
        for i in range(1, n + 1):
            # Is task 'i' currently running at time 't'?
            if start_times[i] <= t < (start_times[i] + durations[i]):
                for k in range(K):
                    current_usage[k] += resource_reqs[i][k]
                    
        # Verify usage doesn't exceed total capacities
        for k in range(K):
            if current_usage[k] > capacities[k]:
                print(f"❌ RESOURCE VIOLATION:")
                print(f"   At time t={t}, Resource {k} usage is {current_usage[k]}, "
                      f"which exceeds the capacity of {capacities[k]}.")
                return False

    print("✅ Resource Constraints: Passed")
    print(f"✅ VALIDATION SUCCESSFUL! Final Makespan: {makespan}")
    print("-----------------------------------\n")
    return True

if __name__ == "__main__":
    solve()