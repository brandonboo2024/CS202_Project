import sys
import time
from collections import deque

class RCPSPSolver:
    def __init__(self, filename, time_limit=30):
        self.filename = filename
        self.time_limit = time_limit
        self.start_time = 0
        self.best_makespan = float('inf')
        self.best_schedule = None
        self.nodes_explored = 0
        self.parse_input()

    def parse_input(self):
        with open(self.filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        parts = lines[0].split()
        self.N = int(parts[0])
        self.R = int(parts[1])

        total_activities = self.N + 2
        self.successors = [[] for _ in range(total_activities)]
        self.predecessors = [[] for _ in range(total_activities)]

        line_idx = 1
        for i in range(total_activities):
            parts = list(map(int, lines[line_idx].split()))
            line_idx += 1
            job_idx = parts[0]
            c = parts[1]
            self.successors[job_idx] = parts[2:2+c]

        self.duration = [0] * total_activities
        self.resources = [[0] * self.R for _ in range(total_activities)]

        for i in range(total_activities):
            parts = list(map(int, lines[line_idx].split()))
            line_idx += 1
            job_idx = parts[0]
            self.duration[job_idx] = parts[1]
            for k in range(self.R):
                self.resources[job_idx][k] = parts[2 + k]

        self.capacity = list(map(int, lines[line_idx].split()))

        for i in range(total_activities):
            for succ in self.successors[i]:
                self.predecessors[succ].append(i)

        self.total_activities = total_activities

    def has_cycle(self):
        visited = [0] * self.total_activities

        def dfs(node):
            visited[node] = 1
            for succ in self.successors[node]:
                if visited[succ] == 1:
                    return True
                if visited[succ] == 0 and dfs(succ):
                    return True
            visited[node] = 2
            return False

        for i in range(self.total_activities):
            if visited[i] == 0:
                if dfs(i):
                    return True
        return False

    def resource_exceeds_capacity(self):
        for i in range(1, self.N + 1):
            for k in range(self.R):
                if self.resources[i][k] > self.capacity[k]:
                    return True
        return False

    def check_infeasibility(self):
        if self.has_cycle():
            return True, "Cycle detected in precedence constraints"
        if self.resource_exceeds_capacity():
            return True, "Activity requires more resources than capacity"
        return False, None

    def critical_path_remaining(self, scheduled, current_time):
        unscheduled = [i for i in range(1, self.N + 1) if not scheduled[i]]
        if not unscheduled:
            return 0

        unscheduled_set = set(unscheduled)
        dist = {act: -float('inf') for act in unscheduled}
        in_degree = {act: 0 for act in unscheduled}
        adj = {act: [] for act in unscheduled}

        for act in unscheduled:
            for succ in self.successors[act]:
                if succ in unscheduled_set:
                    adj[act].append(succ)
                    in_degree[succ] += 1

        queue = deque()
        for act in unscheduled:
            if not any(p in unscheduled_set for p in self.predecessors[act]):
                dist[act] = self.duration[act]
                queue.append(act)

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                dist[v] = max(dist[v], dist[u] + self.duration[v])
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        return max(dist.values()) if dist else 0

    def get_earliest_start(self, finish_times, activity, resource_usage):
        t = max((finish_times[pred] for pred in self.predecessors[activity]), default=0)
        d = self.duration[activity]

        if d == 0:
            return t

        while True:
            while t + d > len(resource_usage):
                resource_usage.append([0] * self.R)

            feasible = True
            for dt in range(d):
                for k in range(self.R):
                    if resource_usage[t + dt][k] + self.resources[activity][k] > self.capacity[k]:
                        feasible = False
                        t = t + dt + 1
                        break
                if not feasible:
                    break

            if feasible:
                return t

    def update_resource_usage(self, resource_usage, activity, start_time, delta):
        d = self.duration[activity]
        while start_time + d > len(resource_usage):
            resource_usage.append([0] * self.R)
        for t in range(start_time, start_time + d):
            for k in range(self.R):
                resource_usage[t][k] += delta * self.resources[activity][k]

    def priority_rule_schedule(self):
        scheduled = [False] * self.total_activities
        scheduled[0] = True
        start_times = [0] * self.total_activities
        finish_times = [0] * self.total_activities
        resource_usage = [[0] * self.R for _ in range(1000)]
        scheduled_count = 0

        while scheduled_count < self.N:
            eligible = []
            for act in range(1, self.N + 1):
                if not scheduled[act]:
                    if all(scheduled[pred] for pred in self.predecessors[act]):
                        eligible.append(act)

            if not eligible:
                return None

            eligible.sort(key=lambda a: self.duration[a], reverse=True)
            act = eligible[0]

            start = self.get_earliest_start(finish_times, act, resource_usage)
            finish = start + self.duration[act]

            start_times[act] = start
            finish_times[act] = finish
            scheduled[act] = True
            scheduled_count += 1
            self.update_resource_usage(resource_usage, act, start, 1)

        makespan = max(finish_times[pred] for pred in self.predecessors[self.N + 1]) if self.predecessors[self.N + 1] else 0

        return {
            'makespan': makespan,
            'start_times': start_times,
            'finish_times': finish_times
        }

    def dfs(self, scheduled, start_times, finish_times, resource_usage, current_time):
        if time.time() - self.start_time > self.time_limit:
            return

        self.nodes_explored += 1

        if all(scheduled[i] for i in range(1, self.N + 1)):
            end_time = max(finish_times[pred] for pred in self.predecessors[self.N + 1]) if self.predecessors[self.N + 1] else 0
            if end_time < self.best_makespan:
                self.best_makespan = end_time
                self.best_schedule = {
                    'makespan': end_time,
                    'start_times': start_times.copy(),
                    'finish_times': finish_times.copy()
                }
            return

        lb = current_time + self.critical_path_remaining(scheduled, current_time)

        for k in range(self.R):
            remaining_work = sum(
                self.duration[act] * self.resources[act][k]
                for act in range(1, self.N + 1)
                if not scheduled[act]
            )
            if remaining_work > 0:
                resource_lb = current_time + (remaining_work + self.capacity[k] - 1) // self.capacity[k]
                lb = max(lb, resource_lb)

        if lb >= self.best_makespan:
            return

        eligible = [
            act for act in range(1, self.N + 1)
            if not scheduled[act]
            and all(scheduled[pred] for pred in self.predecessors[act])
        ]

        eligible.sort(key=lambda a: len(self.successors[a]), reverse=True)

        for act in eligible:
            start = self.get_earliest_start(finish_times, act, resource_usage)
            finish = start + self.duration[act]

            old_start = start_times[act]
            old_finish = finish_times[act]
            start_times[act] = start
            finish_times[act] = finish
            scheduled[act] = True
            self.update_resource_usage(resource_usage, act, start, 1)

            self.dfs(scheduled, start_times, finish_times, resource_usage, max(current_time, finish))

            scheduled[act] = False
            start_times[act] = old_start
            finish_times[act] = old_finish
            self.update_resource_usage(resource_usage, act, start, -1)

    def solve(self):
        infeasible, reason = self.check_infeasibility()
        if infeasible:
            return None

        self.start_time = time.time()

        initial_schedule = self.priority_rule_schedule()
        if initial_schedule is None:
            return None

        self.best_makespan = initial_schedule['makespan']
        self.best_schedule = initial_schedule
        
        scheduled = [False] * self.total_activities
        scheduled[0] = True
        start_times = [0] * self.total_activities
        finish_times = [0] * self.total_activities
        resource_usage = [[0] * self.R for _ in range(2000)]

        self.dfs(scheduled, start_times, finish_times, resource_usage, 0)

        return self.best_schedule


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("-1")
        sys.exit(1)

    solver = RCPSPSolver(sys.argv[1], time_limit=30)
    solution = solver.solve()

    if solution is None or solution['makespan'] == float('inf'):
        print("-1")
    else:
        print(",".join(str(solution['start_times'][i]) for i in range(1, solver.N + 1)))