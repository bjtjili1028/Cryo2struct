// @author: nabin
// conda install -c conda-forge gxx  -> if older version of g++ is present
// g++ -fPIC -shared -o viterbi.so viterbi.cpp -O3

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <tuple> 
#include <set>

// 全局變數，用來指示是否已成功運算完畢
bool success = false;

// 全局排除狀態容器：存放需要排除（不處理）的狀態索引
std::vector<int> exclude_states_in_c; //global exclude states

// 以下是 viterbi 函數以及 run_viterbi 函數的 C 語言介面宣告，使其可以供其他語言（例如 Python）調用

// 使用 extern "C" 指定 C 語言鏈接方式，避免 C++ 名稱改編問題
// Function prototype for viterbi_main with C linkage specification
extern "C" void run_viterbi(std::vector<int> observation, int num_observations, 
                            std::vector<int> states, int num_states, 
                            std::vector<std::vector<double>> transition_matrix, 
                            std::vector<std::vector<double>> emission_matrix, 
                            std::vector<double> initial_matrix, 
                            std::vector<int> states_to_work_python);


// viterbi 函數：利用 Viterbi 算法計算最佳隱藏狀態序列，並根據條件動態地排除部分狀態
// Implementation of viterbi with C linkage specification
extern "C" void viterbi(std::vector<int> observation, int num_observations, 
                        std::vector<int> states, int num_states, 
                        std::vector<std::vector<double>> transition_matrix, 
                        std::vector<std::vector<double>> emission_matrix, 
                        std::vector<double> initial_matrix, 
                        std::vector<int> states_to_work_python) 
{   // 建立 trellis 矩陣，大小為 (num_observations x num_states)
    // 用來保存在各個時間點上，每個狀態的最大概率累計值
    // Run viterbi program
    std::vector<std::vector<double>> trellis(num_observations, std::vector<double>(num_states, 0));
    
    // 用 unordered_map 儲存各狀態的最佳路徑（每個狀態對應到一個 state 序列）
    std::unordered_map<int, std::vector<int>> path;  //an empty unordered_map
    
    // 建立實際用於計算的狀態集，過濾掉全局排除清單中的狀態
    std::vector<int> states_to_work_with;
    for (int x=0; x<states_to_work_python.size(); x++)
    {   
        // 如果當前狀態不在全局排除清單中，則加入處理列表
        if (std::find(exclude_states_in_c.begin(), exclude_states_in_c.end(), states_to_work_python[x]) == exclude_states_in_c.end())
        {
            states_to_work_with.push_back(states_to_work_python[x]);
        }
    }
    // 初始化第一個觀測值的機率值與路徑
    for (int state: states_to_work_with)
    {   
        // 初始概率 = 初始狀態概率 + 第一個觀測值的發射概率
        trellis[0][state] = initial_matrix[state] + emission_matrix[state][observation[0]];
        // 初始路徑僅包含自身狀態
        path[state].push_back(state);
    }

    // 依序計算後續各個觀測點的狀態機率
    for (int observation_index=1; observation_index<num_observations; observation_index++)
    {   
        // 為每個時間點建立新的路徑儲存結構
        std::unordered_map<int, std::vector<int>> new_path;
        
        // 對於所有可處理的狀態進行遍歷
        for (int state: states_to_work_with)
        {
            double max_prob = -std::numeric_limits<float>::infinity();
            int possible_state = -1; // 保存使機率最大的上一個狀態
            
            // 遍歷所有上一個時間點可行的狀態，嘗試找到最佳轉移
            for (int previous_state: states_to_work_with)
            {
                // 過濾掉已在上一個路徑中出現的狀態，避免重複
                auto it = std::find(path[previous_state].begin(), path[previous_state].end(), state);
                if (it != path[previous_state].end())
                {
                    continue;
                }
                // 計算從 previous_state 到當前 state 的轉移概率累積值，加上當前觀測的發射概率
                double prob = trellis[observation_index - 1][previous_state] + transition_matrix[previous_state][state] + emission_matrix[state][observation[observation_index]];
                if (prob > max_prob)
                {
                    max_prob = prob;
                    possible_state = previous_state;
                }
            }
            double probability = max_prob;
            // 若未找到合適的上一狀態（possible_state 為 -1），則進行特殊處理
            if (possible_state == -1)
            {
                // 如果已經成功則直接返回
                if (success == true){return;}
                
                // 否則，找出上一層中概率最大的狀態作為替代
                auto max_it = std::max_element(states_to_work_with.begin(), states_to_work_with.end(),
                               [&](const auto& state1, const auto& state2) {
                                   return trellis[observation_index - 1][state1] <
                                          trellis[observation_index - 1][state2];
                               });

                auto probability = trellis[observation_index - 1][*max_it];
                auto state = *max_it;
                
                // 將該狀態的路徑中的所有狀態加入全局排除清單中，避免重複使用
                exclude_states_in_c.insert(exclude_states_in_c.end(), path[state].begin(), path[state].end());
                // 形成剩餘的觀測序列（從當前索引到最後）
                std::vector<int> sub_observation(observation.begin() + observation_index, observation.end());
                // 遞迴呼叫 run_viterbi 處理剩餘部分
                run_viterbi(sub_observation, sub_observation.size(), states, num_states, transition_matrix, emission_matrix, initial_matrix, states_to_work_python);
            }
            // 將當前狀態在此時間點的累計概率更新到 trellis 中
            trellis[observation_index][state] = probability;
            // 新路徑為從最佳上一個狀態的路徑加上當前狀態
            new_path[state] = path[possible_state];
            new_path[state].push_back(state);
        }
        // 更新路徑結構為最新計算結果
        path = new_path;
    }
    // 最後，在所有狀態中找出最終時間點擁有最大累計概率的狀態
    auto max_it = std::max_element(states_to_work_with.begin(), states_to_work_with.end(),
                    [&](const auto& state1, const auto& state2) {
                        return trellis[num_observations - 1][state1] <
                                trellis[num_observations - 1][state2];
                    });
    auto probability = trellis[num_observations - 1][*max_it];
    auto state = *max_it;   
    // 將該狀態的最佳路徑加入全局排除清單中，避免後續重複使用
    exclude_states_in_c.insert(exclude_states_in_c.end(), path[state].begin(), path[state].end());
    
    // 設置成功標誌，表示 Viterbi 演算法已順利完成
    success = true;
    return;
    
}

// run_viterbi 為包裝函數，其內部直接調用 viterbi 函數
extern "C" void run_viterbi(std::vector<int> observation, int num_observations, std::vector<int> states, int num_states, std::vector<std::vector<double>> transition_matrix, std::vector<std::vector<double>> emission_matrix, std::vector<double> initial_matrix, std::vector<int> states_to_work_python)
{
    viterbi(observation, num_observations, states, num_states, transition_matrix, emission_matrix, initial_matrix, states_to_work_python); 
    return;
    
}

// 以下為 viterbi_main 函數，提供 C 語言介面，方便從外部（例如 Python）以指標方式傳入資料
// Implementation of viterbi_main with C linkage specification
extern "C" int* viterbi_main(int* obs, int num_observations, int num_states, double* transt, double* emiss, double* init, int* exclude_s, int exclude_s_len) 
{
    // 將成功旗標設為 false，等待運算完成後更新
    success = false;
    
    // 建立並初始化觀測值向量
    std::vector<int> observations(num_observations);
    // 建立並初始化狀態向量，此處假設狀態用索引表示（0,1,2,...,num_states-1）
    std::vector<int> states(num_states);
    
    // 初始化狀態轉移矩陣：行與列均為 num_states
    std::vector<std::vector<double>> transition_matrix(num_states, std::vector<double>(num_states));
    // 初始化發射矩陣：每個狀態對應 20 個可能的發射值
    std::vector<std::vector<double>> emission_matrix(num_states, std::vector<double>(20));
    // 初始化初始狀態概率向量
    std::vector<double> initial_matrix(num_states);
    
    // 將從外部傳入的排除狀態資料存入 exclude_stat 向量中
    std::vector<int> exclude_stat(exclude_s_len);

    // 根據傳入的指標資料初始化初始概率矩陣與轉移矩陣
    for (int i = 0; i<num_states; i++)
    {
        initial_matrix[i] = init[i];
        states[i] = i;
        for (int j = 0; j<num_states; j++)
        {
            transition_matrix[i][j] = transt[i*num_states + j];
        }
    }
    
    // 根據傳入的指標資料初始化發射矩陣（每個狀態20個可能發射值）
    for (int i = 0; i<num_states; i++)
    {
        for (int j = 0; j<20; j++)
        {
            emission_matrix[i][j] = emiss[i*20 + j];
        }
    }

    // 將排除狀態資料填入 exclude_stat 向量
    for (int i = 0; i<exclude_s_len; i++)
    {
        exclude_stat[i] = exclude_s[i];
    }

    // 將外部傳入的觀測值陣列賦值到 observations 向量中
    for (int i = 0; i<num_observations; i++)
    {
        observations[i] = obs[i];
    }

    // 構造用於 Viterbi 計算的狀態工作集，過濾掉需要排除的狀態
    std::vector<int> states_to_work_python;
    for (int x=0; x<num_states; x++)
    {
        if (std::find(exclude_stat.begin(), exclude_stat.end(), states[x]) == exclude_stat.end())
        {
            states_to_work_python.push_back(states[x]);
        }
    }
    
    // 清空全局排除清單，確保每次計算前排除列表均為空
    exclude_states_in_c.clear();

    // 呼叫 run_viterbi 開始運算
    run_viterbi(observations, num_observations, states, num_states, transition_matrix, emission_matrix, initial_matrix, states_to_work_python);
    
    // 將全局排除狀態轉換成 set (用以去重，但下方返回的是 vector 的 data pointer)
    std::set<int> s(exclude_states_in_c.begin(), exclude_states_in_c.end());

    // 返回全局排除狀態的內部資料指標，注意在跨語言邊界下要小心記憶體管理
    int * result = exclude_states_in_c.data();
    return result;
    
}
