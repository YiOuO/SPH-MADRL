#include "TrainSrc/include/DRLPipline.h"

DRLPipline::DRLPipline(int num_agents_, std::vector<int> local_obs_dim, std::vector<int> local_act_dim,
                       int policy_freq, int buffer_size, float alpha, float alpha_lr,
                       float lr, float tau, float gamma, bool use_cuda)
{
    this->num_agents = num_agents_;
    my_param = std::make_shared<DRLParam>();

    my_multi_agents = std::make_shared<MultiAgents>(num_agents_, local_obs_dim, local_act_dim,
                                                    alpha, alpha_lr, gamma, tau, lr, policy_freq, use_cuda);

    // Size the replay buffer using the combined agent dimensions.
    const int state_dim  = std::accumulate(local_obs_dim.begin(), local_obs_dim.end(), 0);
    const int action_dim = std::accumulate(local_act_dim.begin(), local_act_dim.end(), 0);
    const int next_state_dim = state_dim;
    const int done_dim = num_agents_;//1？
    const int reward_dim = num_agents_;
    my_replay_buffer = std::make_shared<ReplayBuffer>(buffer_size, state_dim, action_dim, next_state_dim, done_dim, reward_dim);

    reward_total.clear();
}

DRLPipline::~DRLPipline()
{
}

void DRLPipline::sampleFromReplayBuffer(int batch_num,
                                        torch::Tensor &state_batch,
                                        torch::Tensor &action_batch,
                                        torch::Tensor &next_state_batch,
                                        torch::Tensor &done_batch,
                                        torch::Tensor &reward_batch)
{
    my_replay_buffer->sample(batch_num,
                             state_batch,
                             action_batch,
                             next_state_batch,
                             done_batch,
                             reward_batch);
}

void DRLPipline::updateNet(const torch::Tensor &state_batch,
                           const torch::Tensor &action_batch,
                           const torch::Tensor &next_state_batch,
                           const torch::Tensor &done_batch,
                           const torch::Tensor &reward_batch)
{
    my_agent->update(state_batch, action_batch, next_state_batch, done_batch, reward_batch);
}

// ✅ DRL vector part

float DRLPipline::getAction(const std::vector<float> &nn_input)
{
    std::vector<float> input = nn_input;
    float output = my_agent->selectAction(input)[0][0].item().toFloat();
    return output;
}

std::vector<float> DRLPipline::getAction(const std::vector<float> &nn_input, bool training)
{

    std::vector<float> input = nn_input;

    torch::Tensor output_tensor = training ? my_agent->selectActionWithNoise(input) : my_agent->selectRandomAction();

    output_tensor = output_tensor.to(torch::kCPU).contiguous();
    std::vector<float> output(output_tensor.data_ptr<float>(),
                              output_tensor.data_ptr<float>() + output_tensor.numel());
    return output;
}

std::vector<std::vector<float>> DRLPipline::getMultiActions(const std::vector<std::vector<float>> &nn_inputs, bool training, bool run_mode)
{
    if ((int)nn_inputs.size() != num_agents)
    {
        std::cout << "[DRLPipline::getMultiActions] inputsize = " << nn_inputs.size() << std::endl;
        throw std::runtime_error("[DRLPipline::getMultiActions] inputs size mismatch");
    }

    std::vector<std::vector<float>> actions(num_agents);
    if(!run_mode){
        // Get one action tensor per agent.
        std::vector<torch::Tensor> output = training
                                                ? my_multi_agents->selectActionsWithNoise(nn_inputs) // N × (1, act_dim_i)
                                                : my_multi_agents->selectRandomActions();

        if ((int)output.size() != num_agents)
        {
            throw std::runtime_error("[DRLPipline::getMultiActions] outputs size mismatch");
        }

        // Convert each action tensor to a float vector.
        for (int i = 0; i < num_agents; ++i)
        {
            auto t = output[i].to(torch::kCPU).contiguous().view({-1}); // Flatten to (act_dim_i).
            const float *p = t.data_ptr<float>();
            actions[i].assign(p, p + t.numel());
        }
    }
    else
    {
        std::vector<torch::Tensor> output = my_multi_agents->selectActionsWithoutNoise(nn_inputs);
        for (int i = 0; i < num_agents; ++i)
        {
            auto t = output[i].to(torch::kCPU).contiguous().view({-1}); // Flatten to (act_dim_i).
            const float *p = t.data_ptr<float>();
            actions[i].assign(p, p + t.numel());
        }      
    }
    return actions;
}

void DRLPipline::storeToReplayBuffer(
    const std::vector<std::vector<float>> &state,      // N × S_i
    const std::vector<std::vector<float>> &action,     // N × A_i
    const std::vector<std::vector<float>> &next_state, // N × S_i
    const std::vector<int> &done,                      // N
    const std::vector<float> &reward)                  // N
{
    auto fopts = torch::TensorOptions().dtype(torch::kFloat32);
    auto iopts = torch::TensorOptions().dtype(torch::kInt32);

    // Flatten per-agent vectors into a (1, sum_dim) tensor.
    auto cat_float_2d = [&](const std::vector<std::vector<float>>& vv) -> torch::Tensor {
        size_t total = 0;
        for (const auto& v : vv) total += v.size();
        std::vector<float> flat; flat.reserve(total);
        for (const auto& v : vv) flat.insert(flat.end(), v.begin(), v.end());
        return torch::from_blob(flat.data(), {1, static_cast<long>(flat.size())}, fopts).clone();
    };

    // Build tensors from the flattened per-agent vectors.
    torch::Tensor state_tensor      = cat_float_2d(state);       // (1, ΣS_i)
    torch::Tensor action_tensor     = cat_float_2d(action);      // (1, ΣA_i)
    torch::Tensor next_state_tensor = cat_float_2d(next_state);  // (1, ΣS_i)

    // Keep per-agent done and reward values in (1, N) tensors.
    torch::Tensor done_tensor   = torch::from_blob(const_cast<int*>(done.data()),
                                 {1, static_cast<long>(done.size())}, fopts).clone();     // (1, N)
    torch::Tensor reward_tensor = torch::from_blob(const_cast<float*>(reward.data()),
                                 {1, static_cast<long>(reward.size())}, fopts).clone();   // (1, N)


    my_replay_buffer->store(state_tensor, action_tensor, next_state_tensor, done_tensor, reward_tensor);
}

void DRLPipline::storeToReplayBuffer(
    const std::vector<float> &state,
    const std::vector<float> &action,
    const std::vector<float> &next_state,
    const int &done,
    const float &reward)
{
    torch::Tensor state_tensor;
    torch::Tensor action_tensor;
    torch::Tensor next_state_tensor;
    torch::Tensor done_tensor;
    torch::Tensor reward_tensor;

    state_tensor = torch::from_blob(
                       const_cast<float *>(state.data()),
                       {1, static_cast<long>(state.size())},
                       torch::TensorOptions().dtype(torch::kFloat32))
                       .clone();


    action_tensor = torch::from_blob(
                        const_cast<float *>(action.data()),
                        {1, static_cast<long>(action.size())},
                        torch::TensorOptions().dtype(torch::kFloat32))
                        .clone();

    next_state_tensor = torch::from_blob(
                            const_cast<float *>(next_state.data()),
                            {1, static_cast<long>(next_state.size())},
                            torch::TensorOptions().dtype(torch::kFloat32))
                            .clone();

    std::vector<int> done_tmp{done};
    done_tensor = torch::from_blob(done_tmp.data(), {1, 1}, torch::TensorOptions().dtype(torch::kInt32)).clone();

    std::vector<float> reward_tmp{reward};
    reward_tensor = torch::from_blob(reward_tmp.data(), {1, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();

    my_replay_buffer->store(state_tensor, action_tensor, next_state_tensor, done_tensor, reward_tensor);
}

std::vector<float> DRLPipline::getMultiRewardAcplate(const int num_agents,const int steps_train
    ,const std::vector<float> & plate_dz, const std::vector<float>& sum_wgreward)
{
    std::vector<float> rewards(num_agents);
    const float gamma = 10.0f, lamda = 0.5f, epsilon = 0.12f; 
    float steps = float(steps_train);
    for (int i = 0; i < num_agents; i++)
    {
        // rewards[i] = gamma*(epsilon*0.08f-fabs(sum_wgreward[i])/steps) - lamda*fabs(plate_dz[i]/0.05f)*fabs(plate_dz[i]/0.05f);
        rewards[i] = -gamma*(fabs(sum_wgreward[i])/steps)/0.05f - lamda*fabs(plate_dz[i]/0.05f)*fabs(plate_dz[i]/0.05f);
    }
    return rewards;
}



std::vector<float> DRLPipline::getMultiRewardWecPto(int num_agents,const std::vector<float> & sum_stepreward)
{
    std::vector<float> rewards(num_agents);
    float baseline = 15.6f;
    float total_reward = 0.0f;
    for (int i = 0; i< num_agents; i++)
    {
        total_reward += sum_stepreward[i];
    }

    for (int i = 0; i < num_agents; i++)
    {
        float reward_others = total_reward-sum_stepreward[i];
        float reward_i = 0.7f * sum_stepreward[i] + 0.3f * reward_others;
        rewards[i] = reward_i/baseline-1.0f;
    }
    return rewards;
}


//- r7
std::vector<float> DRLPipline::getMultiRewardWecPto(
    int num_agents,
    const std::vector<float>& sum_stepreward,
    const std::vector<float>& last_pto,
    const std::vector<float>& value_pto)
{
    std::vector<float> rewards(num_agents);
    float baseline = 0.0f;
    float total_reward = 0.0f;

    // Compute the mean reward across agents.
    for (int i = 0; i < num_agents; i++) {
        total_reward += sum_stepreward[i];
    }
    float avg_reward = total_reward / num_agents;

    // Combine local and global rewards.
    float alpha = 0.3f;  // Weight of the local reward.
    for (int i = 0; i < num_agents; i++) {
        // Blend each local reward with the mean reward.
        float r = alpha * sum_stepreward[i] + (1.0f - alpha) * avg_reward;

        float penalty = 0.0f;

        rewards[i] = r + penalty - baseline;
    }

    return rewards;
}



void DRLPipline::writeTotalRewardToFile(std::string reward_filename, bool first_flag)
{
    std::ofstream reward_file;

    if (first_flag)
    {
        reward_file.open(reward_filename, std::ios_base::out);
    }
    else
    {
        reward_file.open(reward_filename, std::ios::app);
    }

    if (reward_file.is_open())
    {
        // for(const float& reward : reward_total){
        //     reward_file << reward << std::endl;
        // }
        if (reward_total.size() > 0)
            reward_file << reward_total.back() << std::endl;

        reward_file.close();
        std::cout << "reward written to file" << std::endl;
    }
    else
    {
        std::cout << "cannot open file" << std::endl;
    }
}

void DRLPipline::updateNet(int batch_size)
{
    torch::Tensor state_batch;
    torch::Tensor action_batch;
    torch::Tensor next_state_batch;
    torch::Tensor done_batch;
    torch::Tensor reward_batch;
    sampleFromReplayBuffer(batch_size,
                           state_batch,
                           action_batch,
                           next_state_batch,
                           done_batch,
                           reward_batch);

    my_agent->update(state_batch, action_batch, next_state_batch, done_batch, reward_batch);
}



void DRLPipline::updateMultiNet(int batch_size,float time)
{
    torch::Tensor state_batch;
    torch::Tensor action_batch;
    torch::Tensor next_state_batch;
    torch::Tensor done_batch;
    torch::Tensor reward_batch;
    sampleFromReplayBuffer(batch_size,
                           state_batch,
                           action_batch,
                           next_state_batch,
                           done_batch,
                           reward_batch);



    my_multi_agents->update(state_batch, action_batch, next_state_batch, done_batch, reward_batch, time);
}


void DRLPipline::saveMultiModels()
{
    std::cout << "save model to " << my_param->save_model_dir << std::endl;
    my_multi_agents->save(my_param->save_model_dir);
}


void DRLPipline::loadMultiModels()
{
    std::cout << "load model from " << my_param->load_model_dir << std::endl;
    my_multi_agents->load(my_param->load_model_dir);
}

void DRLPipline::broadcastSingleModel()
{
    std::cout << "broadcast single model from " 
              << my_param->load_model_dir << " to all agents" << std::endl;
    my_multi_agents->loadSingleModelToAll(my_param->load_model_dir);
}
