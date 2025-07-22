// 修正版本的 viterbi.cpp，保留原始邏輯，加入防錯機制避免 segmentation fault

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <tuple> 
#include <set>

bool success = false;
const int VITERBI_RECURSION_LIMIT = 100;
static int viterbi_recursion_depth = 0;

std::vector<int> exclude_states_in_c; // global exclude states

extern "C" void run_viterbi(std::vector<int> observation, int num_observations, std::vector<int> states, int num_states,
                             std::vector<std::vector<double>> transition_matrix,
                             std::vector<std::vector<double>> emission_matrix,
                             std::vector<double> initial_matrix, std::vector<int> states_to_work_python);

extern "C" void viterbi(std::vector<int> observation, int num_observations, std::vector<int> states, int num_states,
                        std::vector<std::vector<double>> transition_matrix,
                        std::vector<std::vector<double>> emission_matrix,
                        std::vector<double> initial_matrix, std::vector<int> states_to_work_python) {
    
    if (viterbi_recursion_depth++ > VITERBI_RECURSION_LIMIT) {
        std::cerr << "[Error] Viterbi recursion exceeded limit.\n";
        viterbi_recursion_depth--;
        return;
    }

    std::vector<std::vector<double>> trellis(num_observations, std::vector<double>(num_states, 0));
    std::unordered_map<int, std::vector<int>> path;
    std::vector<int> states_to_work_with;

    for (int x : states_to_work_python) {
        if (std::find(exclude_states_in_c.begin(), exclude_states_in_c.end(), x) == exclude_states_in_c.end()) {
            states_to_work_with.push_back(x);
        }
    }

    if (observation.empty() || states_to_work_with.empty()) {
        viterbi_recursion_depth--;
        return;
    }

    for (int state : states_to_work_with) {
        if (observation[0] >= emission_matrix[state].size()) continue;
        trellis[0][state] = initial_matrix[state] + emission_matrix[state][observation[0]];
        path[state].push_back(state);
    }

    for (int observation_index = 1; observation_index < num_observations; observation_index++) {
        std::unordered_map<int, std::vector<int>> new_path;
        for (int state : states_to_work_with) {
            double max_prob = -std::numeric_limits<float>::infinity();
            int possible_state = -1;
            for (int previous_state : states_to_work_with) {
                if (path.find(previous_state) == path.end()) continue;
                if (std::find(path[previous_state].begin(), path[previous_state].end(), state) != path[previous_state].end()) continue;
                if (observation[observation_index] >= emission_matrix[state].size()) continue;

                double prob = trellis[observation_index - 1][previous_state] +
                              transition_matrix[previous_state][state] +
                              emission_matrix[state][observation[observation_index]];
                if (prob > max_prob) {
                    max_prob = prob;
                    possible_state = previous_state;
                }
            }

            if (possible_state == -1 || path.find(possible_state) == path.end()) {
                if (success) {
                    viterbi_recursion_depth--;
                    return;
                }
                auto max_it = std::max_element(states_to_work_with.begin(), states_to_work_with.end(),
                                               [&](int a, int b) {
                                                   return trellis[observation_index - 1][a] < trellis[observation_index - 1][b];
                                               });
                int state = *max_it;
                exclude_states_in_c.insert(exclude_states_in_c.end(), path[state].begin(), path[state].end());
                std::vector<int> sub_observation(observation.begin() + observation_index, observation.end());
                run_viterbi(sub_observation, sub_observation.size(), states, num_states, transition_matrix, emission_matrix, initial_matrix, states_to_work_python);
                viterbi_recursion_depth--;
                return;
            }

            trellis[observation_index][state] = max_prob;
            new_path[state] = path[possible_state];
            new_path[state].push_back(state);
        }
        path = new_path;
    }

    auto max_it = std::max_element(states_to_work_with.begin(), states_to_work_with.end(),
                                   [&](int a, int b) {
                                       return trellis[num_observations - 1][a] < trellis[num_observations - 1][b];
                                   });
    int state = *max_it;
    exclude_states_in_c.insert(exclude_states_in_c.end(), path[state].begin(), path[state].end());
    success = true;
    viterbi_recursion_depth--;
    return;
}

extern "C" void run_viterbi(std::vector<int> observation, int num_observations, std::vector<int> states, int num_states,
                             std::vector<std::vector<double>> transition_matrix,
                             std::vector<std::vector<double>> emission_matrix,
                             std::vector<double> initial_matrix, std::vector<int> states_to_work_python) {
    viterbi(observation, num_observations, states, num_states, transition_matrix, emission_matrix, initial_matrix, states_to_work_python);
}

extern "C" int* viterbi_main(int* obs, int num_observations, int num_states, double* transt, double* emiss, double* init, int* exclude_s, int exclude_s_len) {
    success = false;
    viterbi_recursion_depth = 0;

    std::vector<int> observations(obs, obs + num_observations);
    std::vector<int> states(num_states);
    std::vector<std::vector<double>> transition_matrix(num_states, std::vector<double>(num_states));
    std::vector<std::vector<double>> emission_matrix(num_states, std::vector<double>(20));
    std::vector<double> initial_matrix(init, init + num_states);
    std::vector<int> exclude_stat(exclude_s, exclude_s + exclude_s_len);

    for (int i = 0; i < num_states; ++i) {
        states[i] = i;
        for (int j = 0; j < num_states; ++j) {
            transition_matrix[i][j] = transt[i * num_states + j];
        }
        for (int j = 0; j < 20; ++j) {
            emission_matrix[i][j] = emiss[i * 20 + j];
        }
    }

    std::vector<int> states_to_work_python;
    for (int x = 0; x < num_states; ++x) {
        if (std::find(exclude_stat.begin(), exclude_stat.end(), states[x]) == exclude_stat.end()) {
            states_to_work_python.push_back(states[x]);
        }
    }

    exclude_states_in_c.clear();
    run_viterbi(observations, num_observations, states, num_states, transition_matrix, emission_matrix, initial_matrix, states_to_work_python);
    int* result = exclude_states_in_c.data();
    return result;
}
