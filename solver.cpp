#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Instance {
    int n = 0;
    int k = 0;
    std::vector<std::vector<int>> successors;
    std::vector<std::vector<int>> predecessors;
    std::vector<int> durations;
    std::vector<std::vector<int>> demands;
    std::vector<int> capacities;

    int num_jobs() const { return n + 2; }
    int sink() const { return n + 1; }
};

struct Features {
    std::vector<int> transitive_successors;
    std::vector<int> bottom;
    std::vector<int> slack;
    std::vector<int> total_demand;
    std::vector<double> weighted_demand;
};

struct Candidate {
    std::vector<int> starts;
    std::vector<int> order;
};

enum class Rule {
    Bottom,
    Successors,
    Demand,
    Slack,
    Longest,
    Shortest,
};

static std::vector<int> parse_ints(const std::string& line) {
    std::istringstream in(line);
    std::vector<int> values;
    int value;
    while (in >> value) {
        values.push_back(value);
    }
    return values;
}

static Instance parse_instance(std::istream& input) {
    std::vector<std::vector<int>> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        rows.push_back(parse_ints(line));
    }
    if (rows.empty() || rows[0].size() != 2) {
        throw std::runtime_error("first line must contain N and R");
    }

    Instance inst;
    inst.n = rows[0][0];
    inst.k = rows[0][1];
    const int total = inst.num_jobs();
    const int expected_rows = 1 + total + total + 1;
    if (static_cast<int>(rows.size()) != expected_rows) {
        throw std::runtime_error("unexpected row count");
    }

    inst.successors.assign(total, {});
    for (int offset = 0; offset < total; ++offset) {
        const auto& row = rows[1 + offset];
        if (row.size() < 2) {
            throw std::runtime_error("malformed successor row");
        }
        const int job = row[0];
        const int count = row[1];
        if (job != offset || count != static_cast<int>(row.size()) - 2) {
            throw std::runtime_error("successor row mismatch");
        }
        for (int idx = 2; idx < static_cast<int>(row.size()); ++idx) {
            const int succ = row[idx];
            if (succ < 0 || succ >= total) {
                throw std::runtime_error("successor out of range");
            }
            inst.successors[job].push_back(succ);
        }
    }

    inst.durations.assign(total, 0);
    inst.demands.assign(total, std::vector<int>(inst.k, 0));
    const int base = 1 + total;
    for (int offset = 0; offset < total; ++offset) {
        const auto& row = rows[base + offset];
        if (static_cast<int>(row.size()) != 2 + inst.k) {
            throw std::runtime_error("malformed duration/resource row");
        }
        const int job = row[0];
        if (job != offset || row[1] < 0) {
            throw std::runtime_error("duration row mismatch");
        }
        inst.durations[job] = row[1];
        for (int r = 0; r < inst.k; ++r) {
            if (row[2 + r] < 0) {
                throw std::runtime_error("negative demand");
            }
            inst.demands[job][r] = row[2 + r];
        }
    }

    inst.capacities = rows.back();
    if (static_cast<int>(inst.capacities.size()) != inst.k) {
        throw std::runtime_error("capacity count mismatch");
    }

    inst.predecessors.assign(total, {});
    for (int i = 0; i < total; ++i) {
        for (int j : inst.successors[i]) {
            inst.predecessors[j].push_back(i);
        }
    }
    return inst;
}

static std::vector<int> topological_order(const Instance& inst) {
    const int total = inst.num_jobs();
    std::vector<int> indegree(total, 0);
    std::vector<int> queue;
    for (int i = 0; i < total; ++i) {
        indegree[i] = static_cast<int>(inst.predecessors[i].size());
        if (indegree[i] == 0) {
            queue.push_back(i);
        }
    }
    std::vector<int> order;
    for (int head = 0; head < static_cast<int>(queue.size()); ++head) {
        const int job = queue[head];
        order.push_back(job);
        for (int succ : inst.successors[job]) {
            --indegree[succ];
            if (indegree[succ] == 0) {
                queue.push_back(succ);
            }
        }
    }
    if (static_cast<int>(order.size()) != total) {
        throw std::runtime_error("precedence graph contains a cycle");
    }
    return order;
}

static Features compute_features(const Instance& inst) {
    const int total = inst.num_jobs();
    const auto order = topological_order(inst);
    std::vector<std::set<int>> successor_sets(total);
    Features features;
    features.transitive_successors.assign(total, 0);
    features.bottom.assign(total, 0);
    features.slack.assign(total, 0);
    features.total_demand.assign(total, 0);
    features.weighted_demand.assign(total, 0.0);

    std::vector<int> earliest(total, 0);
    for (int job : order) {
        const int finish = earliest[job] + inst.durations[job];
        for (int succ : inst.successors[job]) {
            earliest[succ] = std::max(earliest[succ], finish);
        }
    }
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int job = *it;
        int best_tail = 0;
        for (int succ : inst.successors[job]) {
            successor_sets[job].insert(succ);
            successor_sets[job].insert(successor_sets[succ].begin(), successor_sets[succ].end());
            best_tail = std::max(best_tail, features.bottom[succ]);
        }
        features.transitive_successors[job] = static_cast<int>(successor_sets[job].size());
        features.bottom[job] = inst.durations[job] + best_tail;
    }

    int critical_path = 0;
    for (int job = 0; job < total; ++job) {
        critical_path = std::max(critical_path, earliest[job] + inst.durations[job]);
    }
    for (int job = 0; job < total; ++job) {
        features.slack[job] = critical_path - features.bottom[job] - earliest[job];
        for (int r = 0; r < inst.k; ++r) {
            features.total_demand[job] += inst.demands[job][r];
            features.weighted_demand[job] += static_cast<double>(inst.demands[job][r]) / std::max(1, inst.capacities[r]);
        }
    }
    return features;
}

static std::tuple<double, int, int, int> priority_key(const Instance& inst, const Features& features, Rule rule, int job) {
    switch (rule) {
        case Rule::Bottom:
            return {features.bottom[job], features.transitive_successors[job], inst.durations[job], -job};
        case Rule::Successors:
            return {features.transitive_successors[job], static_cast<int>(inst.successors[job].size()), features.bottom[job], -job};
        case Rule::Demand:
            return {features.weighted_demand[job], features.bottom[job], inst.durations[job], -job};
        case Rule::Slack:
            return {-features.slack[job], features.bottom[job], features.transitive_successors[job], -job};
        case Rule::Longest:
            return {inst.durations[job], features.bottom[job], -job, 0};
        case Rule::Shortest:
            return {-inst.durations[job], features.bottom[job], -job, 0};
    }
    return {features.bottom[job], -job, 0, 0};
}

static bool can_place(const Instance& inst, std::vector<std::vector<int>>& usage, int job, int start) {
    for (int t = start; t < start + inst.durations[job]; ++t) {
        if (t >= static_cast<int>(usage.size())) {
            usage.resize(t + inst.durations[job] + 1, std::vector<int>(inst.k, 0));
        }
        for (int r = 0; r < inst.k; ++r) {
            if (usage[t][r] + inst.demands[job][r] > inst.capacities[r]) {
                return false;
            }
        }
    }
    return true;
}

static void place(const Instance& inst, std::vector<std::vector<int>>& usage, int job, int start, int sign = 1) {
    for (int t = start; t < start + inst.durations[job]; ++t) {
        for (int r = 0; r < inst.k; ++r) {
            usage[t][r] += sign * inst.demands[job][r];
        }
    }
}

static int duration_sum(const Instance& inst) {
    return std::accumulate(inst.durations.begin(), inst.durations.end(), 0);
}

static int project_makespan(const Instance& inst, const std::vector<int>& starts) {
    int horizon = 0;
    for (int job = 0; job < inst.num_jobs(); ++job) {
        horizon = std::max(horizon, starts[job] + inst.durations[job]);
    }
    return horizon;
}

static std::vector<std::vector<int>> usage_from_schedule(const Instance& inst, const std::vector<int>& starts) {
    const int horizon = project_makespan(inst, starts);
    std::vector<std::vector<int>> usage(std::max(1, horizon + 1), std::vector<int>(inst.k, 0));
    for (int job = 0; job < inst.num_jobs(); ++job) {
        place(inst, usage, job, starts[job]);
    }
    return usage;
}

static std::vector<int> right_justify(const Instance& inst, const std::vector<int>& starts) {
    const int horizon = project_makespan(inst, starts);
    auto improved = starts;
    auto usage = usage_from_schedule(inst, improved);
    std::vector<int> order(inst.num_jobs());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (improved[a] != improved[b]) {
            return improved[a] > improved[b];
        }
        return a > b;
    });
    for (int job : order) {
        if (job == 0 || job == inst.sink()) {
            continue;
        }
        place(inst, usage, job, improved[job], -1);
        int latest = horizon - inst.durations[job];
        for (int succ : inst.successors[job]) {
            latest = std::min(latest, improved[succ] - inst.durations[job]);
        }
        int earliest = 0;
        for (int pred : inst.predecessors[job]) {
            earliest = std::max(earliest, improved[pred] + inst.durations[pred]);
        }
        int placed = improved[job];
        for (int candidate = latest; candidate >= earliest; --candidate) {
            if (can_place(inst, usage, job, candidate)) {
                placed = candidate;
                break;
            }
        }
        improved[job] = placed;
        place(inst, usage, job, placed);
    }
    return improved;
}

static std::vector<int> left_justify(const Instance& inst, const std::vector<int>& starts) {
    std::vector<int> improved(inst.num_jobs(), -1);
    std::vector<std::vector<int>> usage(std::max(1, starts[inst.sink()] + 1), std::vector<int>(inst.k, 0));
    std::vector<int> order(inst.num_jobs());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (starts[a] != starts[b]) {
            return starts[a] < starts[b];
        }
        return a < b;
    });
    for (int job : order) {
        int earliest = 0;
        for (int pred : inst.predecessors[job]) {
            earliest = std::max(earliest, improved[pred] + inst.durations[pred]);
        }
        int candidate = earliest;
        while (!can_place(inst, usage, job, candidate)) {
            ++candidate;
        }
        improved[job] = candidate;
        place(inst, usage, job, candidate);
    }
    return improved;
}

static std::vector<int> forward_backward_improve(const Instance& inst, const std::vector<int>& starts) {
    return left_justify(inst, right_justify(inst, starts));
}

static Candidate serial_sgs(const Instance& inst, const Features& features, Rule rule, std::mt19937& rng, bool randomize) {
    const int total = inst.num_jobs();
    std::vector<char> scheduled(total, 0);
    std::vector<int> starts(total, -1);
    std::vector<int> activity_order;
    starts[0] = 0;
    scheduled[0] = 1;
    activity_order.push_back(0);
    int remaining = total - 1;
    std::vector<std::vector<int>> usage(std::max(1, duration_sum(inst) + 1), std::vector<int>(inst.k, 0));

    while (remaining > 0) {
        std::vector<int> eligible;
        for (int job = 0; job < total; ++job) {
            if (scheduled[job]) {
                continue;
            }
            bool ok = true;
            for (int pred : inst.predecessors[job]) {
                if (starts[pred] < 0) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                eligible.push_back(job);
            }
        }
        std::sort(eligible.begin(), eligible.end(), [&](int a, int b) {
            return priority_key(inst, features, rule, a) > priority_key(inst, features, rule, b);
        });
        int job = eligible.front();
        if (randomize && eligible.size() > 1) {
            const int top = std::min<int>(4, eligible.size());
            std::vector<int> weights(top);
            for (int i = 0; i < top; ++i) {
                weights[i] = top - i;
            }
            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            job = eligible[dist(rng)];
        }

        int start = 0;
        for (int pred : inst.predecessors[job]) {
            start = std::max(start, starts[pred] + inst.durations[pred]);
        }
        const int max_start = duration_sum(inst) + 1;
        while (!can_place(inst, usage, job, start)) {
            ++start;
            if (start > max_start) {
                throw std::runtime_error("serial SGS exceeded horizon");
            }
        }
        starts[job] = start;
        place(inst, usage, job, start);
        scheduled[job] = 1;
        activity_order.push_back(job);
        --remaining;
    }
    return {starts, activity_order};
}

static std::vector<int> activity_list_sgs(const Instance& inst, const std::vector<int>& activity_order) {
    const int total = inst.num_jobs();
    std::vector<int> position(total, 0);
    for (int i = 0; i < total; ++i) {
        position[activity_order[i]] = i;
    }
    std::vector<char> scheduled(total, 0);
    std::vector<int> starts(total, -1);
    starts[0] = 0;
    scheduled[0] = 1;
    int remaining = total - 1;
    std::vector<std::vector<int>> usage(std::max(1, duration_sum(inst) + 1), std::vector<int>(inst.k, 0));

    while (remaining > 0) {
        int best_job = -1;
        for (int job = 0; job < total; ++job) {
            if (scheduled[job]) {
                continue;
            }
            bool ok = true;
            for (int pred : inst.predecessors[job]) {
                if (starts[pred] < 0) {
                    ok = false;
                    break;
                }
            }
            if (ok && (best_job < 0 || position[job] < position[best_job])) {
                best_job = job;
            }
        }
        int start = 0;
        for (int pred : inst.predecessors[best_job]) {
            start = std::max(start, starts[pred] + inst.durations[pred]);
        }
        const int max_start = duration_sum(inst) + 1;
        while (!can_place(inst, usage, best_job, start)) {
            ++start;
            if (start > max_start) {
                throw std::runtime_error("activity-list SGS exceeded horizon");
            }
        }
        starts[best_job] = start;
        place(inst, usage, best_job, start);
        scheduled[best_job] = 1;
        --remaining;
    }
    return starts;
}

static std::vector<int> perturb_activity_order(const std::vector<int>& activity_order, std::mt19937& rng) {
    auto mutated = activity_order;
    if (mutated.size() <= 3) {
        return mutated;
    }
    std::uniform_int_distribution<int> move_dist(1, 5);
    std::uniform_int_distribution<int> index_dist(1, static_cast<int>(mutated.size()) - 2);
    std::uniform_real_distribution<double> real_dist(0.0, 1.0);
    const int moves = move_dist(rng);
    for (int move = 0; move < moves; ++move) {
        int i = index_dist(rng);
        int j = index_dist(rng);
        if (i == j) {
            continue;
        }
        if (real_dist(rng) < 0.5) {
            std::swap(mutated[i], mutated[j]);
        } else {
            const int job = mutated[i];
            mutated.erase(mutated.begin() + i);
            mutated.insert(mutated.begin() + j, job);
        }
    }
    return mutated;
}

static bool fits_resources(const Instance& inst, const std::vector<int>& used, int job) {
    for (int r = 0; r < inst.k; ++r) {
        if (used[r] + inst.demands[job][r] > inst.capacities[r]) {
            return false;
        }
    }
    return true;
}

static Candidate parallel_sgs(const Instance& inst, const Features& features, Rule rule, std::mt19937& rng, bool randomize) {
    const int total = inst.num_jobs();
    std::vector<int> starts(total, -1);
    std::vector<char> scheduled(total, 0);
    std::vector<int> activity_order;
    starts[0] = 0;
    scheduled[0] = 1;
    activity_order.push_back(0);
    int scheduled_count = 1;
    int t = 0;
    const int horizon = duration_sum(inst) + 1;

    while (scheduled_count < total) {
        std::vector<int> used(inst.k, 0);
        for (int job = 0; job < total; ++job) {
            if (scheduled[job] && starts[job] <= t && t < starts[job] + inst.durations[job]) {
                for (int r = 0; r < inst.k; ++r) {
                    used[r] += inst.demands[job][r];
                }
            }
        }

        bool started_any = false;
        while (true) {
            std::vector<int> eligible;
            for (int job = 0; job < total; ++job) {
                if (scheduled[job]) {
                    continue;
                }
                bool ok = true;
                for (int pred : inst.predecessors[job]) {
                    if (starts[pred] < 0 || starts[pred] + inst.durations[pred] > t) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    eligible.push_back(job);
                }
            }
            if (eligible.empty()) {
                break;
            }
            std::sort(eligible.begin(), eligible.end(), [&](int a, int b) {
                return priority_key(inst, features, rule, a) > priority_key(inst, features, rule, b);
            });
            std::vector<int> candidates = eligible;
            if (randomize && eligible.size() > 1) {
                const int top = std::min<int>(5, eligible.size());
                candidates.assign(eligible.begin(), eligible.begin() + top);
                std::shuffle(candidates.begin(), candidates.end(), rng);
            }
            int chosen = -1;
            for (int job : candidates) {
                if (fits_resources(inst, used, job)) {
                    chosen = job;
                    break;
                }
            }
            if (chosen < 0) {
                break;
            }
            starts[chosen] = t;
            scheduled[chosen] = 1;
            activity_order.push_back(chosen);
            ++scheduled_count;
            for (int r = 0; r < inst.k; ++r) {
                used[r] += inst.demands[chosen][r];
            }
            started_any = true;
        }

        if (scheduled_count >= total) {
            break;
        }
        int next_finish = horizon + 1;
        for (int job = 0; job < total; ++job) {
            if (scheduled[job]) {
                const int finish = starts[job] + inst.durations[job];
                if (finish > t) {
                    next_finish = std::min(next_finish, finish);
                }
            }
        }
        if (next_finish == horizon + 1) {
            if (started_any) {
                continue;
            }
            ++t;
        } else {
            t = next_finish;
        }
        if (t > horizon) {
            throw std::runtime_error("parallel SGS exceeded horizon");
        }
    }
    return {starts, activity_order};
}

static bool impossible_jobs(const Instance& inst) {
    for (int job = 0; job < inst.num_jobs(); ++job) {
        for (int r = 0; r < inst.k; ++r) {
            if (inst.demands[job][r] > inst.capacities[r]) {
                return true;
            }
        }
    }
    return false;
}

static std::vector<int> search_worker(
    const Instance& inst,
    const Features& features,
    Clock::time_point deadline,
    unsigned seed
) {
    std::mt19937 rng(seed);
    const std::vector<Rule> rules = {
        Rule::Bottom, Rule::Successors, Rule::Demand, Rule::Slack, Rule::Longest, Rule::Shortest
    };
    std::vector<int> best;
    std::vector<int> best_order;
    int best_makespan = 1'000'000'000;

    auto consider = [&](std::vector<int> starts, const std::vector<int>& order) {
        starts = forward_backward_improve(inst, starts);
        const int finish = project_makespan(inst, starts);
        if (finish < best_makespan) {
            best_makespan = finish;
            best = std::move(starts);
            if (!order.empty()) {
                best_order = order;
            }
        }
    };

    for (Rule rule : rules) {
        if (Clock::now() >= deadline) {
            break;
        }
        Candidate serial = serial_sgs(inst, features, rule, rng, false);
        consider(serial.starts, serial.order);
        if (Clock::now() >= deadline) {
            break;
        }
        Candidate parallel = parallel_sgs(inst, features, rule, rng, false);
        consider(parallel.starts, parallel.order);
    }

    std::uniform_real_distribution<double> real_dist(0.0, 1.0);
    std::uniform_int_distribution<int> rule_dist(0, static_cast<int>(rules.size()) - 1);
    while (Clock::now() < deadline) {
        const double draw = real_dist(rng);
        if (!best_order.empty() && draw < 0.5) {
            auto order = perturb_activity_order(best_order, rng);
            consider(activity_list_sgs(inst, order), order);
            if (Clock::now() < deadline && real_dist(rng) < 0.2) {
                auto order2 = perturb_activity_order(order, rng);
                consider(activity_list_sgs(inst, order2), order2);
            }
        } else if (draw < 0.8) {
            Candidate parallel = parallel_sgs(inst, features, rules[rule_dist(rng)], rng, true);
            consider(parallel.starts, parallel.order);
        } else {
            Candidate serial = serial_sgs(inst, features, rules[rule_dist(rng)], rng, true);
            consider(serial.starts, serial.order);
        }
    }
    return best;
}

static std::vector<int> solve(const Instance& inst, double time_limit_seconds, int worker_count) {
    if (impossible_jobs(inst)) {
        return {};
    }

    worker_count = std::max(1, worker_count);
    const auto budget = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(std::max(0.001, time_limit_seconds))
    );
    const auto deadline = Clock::now() + budget;
    const Features features = compute_features(inst);
    if (worker_count == 1) {
        return search_worker(inst, features, deadline, 1);
    }

    std::vector<std::vector<int>> worker_bests(worker_count);
    std::vector<std::thread> threads;
    threads.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        threads.emplace_back([&, worker]() {
            worker_bests[worker] = search_worker(inst, features, deadline, 1u + 9973u * static_cast<unsigned>(worker));
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    std::vector<int> best;
    int best_makespan = 1'000'000'000;
    for (auto& candidate : worker_bests) {
        if (!candidate.empty() && project_makespan(inst, candidate) < best_makespan) {
            best_makespan = project_makespan(inst, candidate);
            best = std::move(candidate);
        }
    }
    return best;
}

static void print_solution(const Instance& inst, const std::vector<int>& starts) {
    if (starts.empty()) {
        std::cout << "-1\n";
        return;
    }
    for (int job = 1; job <= inst.n; ++job) {
        if (job > 1) {
            std::cout << ",";
        }
        std::cout << starts[job];
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {
    try {
        Instance inst;
        if (argc >= 2) {
            std::ifstream file(argv[1]);
            if (!file) {
                throw std::runtime_error("could not open input file");
            }
            inst = parse_instance(file);
        } else {
            inst = parse_instance(std::cin);
        }
        const double time_limit = argc >= 3 ? std::stod(argv[2]) : 29.0;
        const int worker_count = argc >= 4 ? std::stoi(argv[3]) : 1;
        print_solution(inst, solve(inst, time_limit, worker_count));
    } catch (...) {
        std::cout << "-1\n";
        return 0;
    }
    return 0;
}
