#include "TrainSrc/include/MultiAgents.h"

// Forward a batch global state to extract the local slice for agent i
torch::Tensor MultiAgents::sliceLocalObservation(const torch::Tensor &batchGlobalState, int agentIndex)
{
    using torch::indexing::Slice;
    const int64_t start = local_obs_offsets[agentIndex];
    const int64_t end = local_obs_offsets[agentIndex + 1];
    return batchGlobalState.index({Slice(), Slice(start, end)});
}

// Concatenate along feature dimension
torch::Tensor MultiAgents::concatAlongFeature(const std::vector<torch::Tensor> &parts)
{
    return torch::cat(parts, /*dim=*/1);
}

std::vector<torch::Tensor>
MultiAgents::selectActionsWithNoise(const std::vector<std::vector<float>> &inputs)
{
    torch::NoGradGuard no_grad;
    if ((int)inputs.size() != num_agents)
    {
        throw std::runtime_error("[MultiAgents::selectActionsWithNoise] inputs size mismatch");
    }

    std::vector<torch::Tensor> actions;
    actions.reserve(num_agents);

    for (int i = 0; i < num_agents; ++i)
    {
        const auto &in_i = inputs[i];
        auto state_i = torch::from_blob(
                           const_cast<float *>(in_i.data()),
                           {1, (long)in_i.size()},
                           torch::TensorOptions().dtype(torch::kFloat32))
                           .clone()
                           .to(my_device);

        // 假设每个体都有自己的 actor[i]，且 sample 返回 (action, log_prob)
        auto [a_i, logp_i] = actors[i]->sample(state_i);     // (1, act_dim_i) on device
        actions.push_back(a_i.to(torch::kCPU).contiguous()); // 方便上层读取指针/转 std::vector
    }
    std::cout << "actions with noises: ";
    for (const auto& a : actions)
    {
        std::cout << a[0].item().toFloat() << " ";
    }
    std::cout << std::endl;
    return actions; // N 个 (1, act_dim_i)
}


std::vector<torch::Tensor>
MultiAgents::selectActionsWithoutNoise(const std::vector<std::vector<float>> &inputs)
{
    torch::NoGradGuard no_grad;

    if ((int)inputs.size() != num_agents)
    {
        throw std::runtime_error("[MultiAgents::selectActionsWithoutNoise] inputs size mismatch");
    }

    std::vector<torch::Tensor> actions;
    actions.reserve(num_agents);

    for (int i = 0; i < num_agents; ++i)
    {
        const auto &in_i = inputs[i];
        auto state_i = torch::from_blob(
                           const_cast<float *>(in_i.data()),
                           {1, static_cast<long>(in_i.size())},
                           torch::TensorOptions().dtype(torch::kFloat32))
                           .clone()
                           .to(my_device);

        auto a_i = actors[i]->predict(state_i); // <-- deterministic output
        actions.push_back(a_i.to(torch::kCPU).contiguous());
    }

    std::cout << "actions without noise : ";
    for (const auto &a : actions)
    {
        std::cout << a[0].item().toFloat() << " ";
    }
    std::cout << std::endl;

    return actions;
}


std::vector<torch::Tensor> MultiAgents::selectRandomActions()
{
    std::vector<torch::Tensor> actions;
    actions.reserve(num_agents);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(my_device);

    for (int i = 0; i < num_agents; ++i)
    {
        // 均匀噪声 [-1, 1], 形状 (1, act_dim_i)
        auto a_i = torch::rand({1, local_act_dim[i]}, opts).mul_(2.0).sub_(1.0);
        actions.push_back(a_i.to(torch::kCPU).contiguous());
    }
    std::cout << "random actions: ";
    for (const auto& a : actions)
    {
        std::cout << a[0].item().toFloat() << " ";
    }
    std::cout << std::endl;
    return actions; // N 个 (1, act_dim_i)
}

// void MultiAgents::update(const torch::Tensor &state_batch_,
//                          const torch::Tensor &action_batch_,
//                          const torch::Tensor &next_state_batch_,
//                          const torch::Tensor &done_batch_,
//                          const torch::Tensor &reward_batch_)
// {
//     // --- device unify ---
//     auto state_batch      = state_batch_.to(my_device);
//     auto action_batch     = action_batch_.to(my_device);
//     auto next_state_batch = next_state_batch_.to(my_device);
//     auto done_batch       = done_batch_.to(my_device);
//     auto reward_batch     = reward_batch_.to(my_device);

//     // --- next actions/logprobs (detach both) ---
//     std::vector<torch::Tensor> nextActionParts(num_agents), nextLogProbParts(num_agents);
//     for (int i = 0; i < num_agents; ++i)
//     {
//         auto next_local_obs_i = sliceLocalObservation(next_state_batch, i);
//         auto [next_act_i, next_logprob_i] = actors[i]->sample(next_local_obs_i);
//         nextActionParts[i]  = next_act_i.detach();
//         nextLogProbParts[i] = next_logprob_i.detach();
//     }
//     auto next_action_batch = concatAlongFeature(nextActionParts); // [B, sumA]

//     // --- global entropy: sum_j alpha_j * logpi_j ---
//     auto weighted_sum_next_logprob = torch::zeros_like(nextLogProbParts[0]); // [B,1]
//     for (int j = 0; j < num_agents; ++j)
//         weighted_sum_next_logprob = weighted_sum_next_logprob + nextLogProbParts[j] * temperatures[j];

//     // --- critics update ---
//     for (int i = 0; i < num_agents; ++i)
//     {
//         torch::Tensor targetQ_i;
//         {
//             torch::NoGradGuard noGrad;
//             auto q1Next = target_critics1[i]->forward(next_state_batch, next_action_batch);
//             auto q2Next = target_critics2[i]->forward(next_state_batch, next_action_batch);
//             auto minQNext = torch::min(q1Next, q2Next);

//             auto reward_i = reward_batch.index({torch::indexing::Slice(), i}).view({-1,1});
//             auto done_i   =   done_batch.index({torch::indexing::Slice(), i}).view({-1,1});

//             targetQ_i = reward_i + gamma * (1.0 - done_i) * (minQNext - weighted_sum_next_logprob);
//         }

//         auto q1 = critics1[i]->forward(state_batch, action_batch);
//         auto q2 = critics2[i]->forward(state_batch, action_batch);
//         auto criticLoss = torch::mse_loss(q1, targetQ_i) + torch::mse_loss(q2, targetQ_i);

//         critic1_optimizers[i]->zero_grad();
//         critic2_optimizers[i]->zero_grad();
//         criticLoss.backward();
//         critic1_optimizers[i]->step();
//         critic2_optimizers[i]->step();
//     }

//     // --- actor + alpha update ---
//     std::vector<torch::Tensor> currentActionParts(num_agents), currentLogProbParts(num_agents);
//     for (int i = 0; i < num_agents; ++i)
//     {
//         auto local_obs_i = sliceLocalObservation(state_batch, i);
//         auto [act_i, logprob_i] = actors[i]->sample(local_obs_i);
//         currentActionParts[i] = act_i;
//         currentLogProbParts[i] = logprob_i;
//     }

//     for (int i = 0; i < num_agents; ++i)
//     {
//         std::vector<torch::Tensor> jointActionParts(num_agents);
//         for (int j = 0; j < num_agents; ++j)
//             jointActionParts[j] = (j == i) ? currentActionParts[j] : currentActionParts[j].detach();

//         auto jointAction = concatAlongFeature(jointActionParts);
//         auto qMin_i = torch::min(
//             critics1[i]->forward(state_batch, jointAction),
//             critics2[i]->forward(state_batch, jointAction));
//         auto actorLoss_i = (temperatures[i] * currentLogProbParts[i] - qMin_i).mean();

//         actor_optimizers[i].zero_grad();
//         actorLoss_i.backward();
//         actor_optimizers[i].step();

//         // - Auto temperature tuning
//         auto alphaLoss_i = -(log_alphas[i] * (currentLogProbParts[i].detach() + target_entropy)).mean();
//         alpha_optimizers[i].zero_grad();
//         alphaLoss_i.backward();
//         alpha_optimizers[i].step();
//         temperatures[i] = log_alphas[i].exp().item<double>();
//     }

//     // --- soft update ---
//     torch::NoGradGuard noGrad;
//     for (int i = 0; i < num_agents; ++i) {
//         soft_update(critics1[i], target_critics1[i]);
//         soft_update(critics2[i], target_critics2[i]);
//     }
// }


void MultiAgents::update(const torch::Tensor &state_batch_,
                         const torch::Tensor &action_batch_,
                         const torch::Tensor &next_state_batch_,
                         const torch::Tensor &done_batch_,
                         const torch::Tensor &reward_batch_,
                         const float time)
{
    // --- device unify ---
    auto state_batch      = state_batch_.to(my_device);
    auto action_batch     = action_batch_.to(my_device);
    auto next_state_batch = next_state_batch_.to(my_device);
    auto done_batch       = done_batch_.to(my_device);
    auto reward_batch     = reward_batch_.to(my_device);

    // --- next actions/logprobs (detach both) ---
    std::vector<torch::Tensor> nextActionParts(num_agents), nextLogProbParts(num_agents);
    for (int i = 0; i < num_agents; ++i)
    {
        auto next_local_obs_i = sliceLocalObservation(next_state_batch, i);
        auto [next_act_i, next_logprob_i] = actors[i]->sample(next_local_obs_i);
        nextActionParts[i]  = next_act_i.detach();
        nextLogProbParts[i] = next_logprob_i.detach();
    }
    auto next_action_batch = concatAlongFeature(nextActionParts); // [B, sumA]

    // --- global entropy: sum_j alpha_j * logpi_j ---
    auto weighted_sum_next_logprob = torch::zeros_like(nextLogProbParts[0]); // [B,1]
    for (int j = 0; j < num_agents; ++j)
        weighted_sum_next_logprob = weighted_sum_next_logprob + nextLogProbParts[j] * temperatures[j];

    // --- critics update ---
    std::vector<double> critic_loss_log(num_agents);  // ← 新增
    for (int i = 0; i < num_agents; ++i)
    {
        torch::Tensor targetQ_i;
        {
            torch::NoGradGuard noGrad;
            auto q1Next = target_critics1[i]->forward(next_state_batch, next_action_batch);
            auto q2Next = target_critics2[i]->forward(next_state_batch, next_action_batch);
            auto minQNext = torch::min(q1Next, q2Next);

            auto reward_i = reward_batch.index({torch::indexing::Slice(), i}).view({-1,1});
            auto done_i   =   done_batch.index({torch::indexing::Slice(), i}).view({-1,1});

            targetQ_i = reward_i + gamma * (1.0 - done_i) * (minQNext - weighted_sum_next_logprob);
        }

        auto q1 = critics1[i]->forward(state_batch, action_batch);
        auto q2 = critics2[i]->forward(state_batch, action_batch);
        auto criticLoss = torch::mse_loss(q1, targetQ_i) + torch::mse_loss(q2, targetQ_i);

        critic1_optimizers[i]->zero_grad();
        critic2_optimizers[i]->zero_grad();
        criticLoss.backward();
        critic1_optimizers[i]->step();
        critic2_optimizers[i]->step();

        critic_loss_log[i] = criticLoss.item<double>();  // ← 新增
    }

    // --- actor + alpha update ---
    std::vector<double> actor_loss_log(num_agents);  // ← 新增

    std::vector<torch::Tensor> currentActionParts(num_agents), currentLogProbParts(num_agents);
    for (int i = 0; i < num_agents; ++i)
    {
        auto local_obs_i = sliceLocalObservation(state_batch, i);
        auto [act_i, logprob_i] = actors[i]->sample(local_obs_i);
        currentActionParts[i] = act_i;
        currentLogProbParts[i] = logprob_i;
    }

    for (int i = 0; i < num_agents; ++i)
    {
        std::vector<torch::Tensor> jointActionParts(num_agents);
        for (int j = 0; j < num_agents; ++j)
            jointActionParts[j] = (j == i) ? currentActionParts[j] : currentActionParts[j].detach();

        auto jointAction = concatAlongFeature(jointActionParts);
        auto qMin_i = torch::min(
            critics1[i]->forward(state_batch, jointAction),
            critics2[i]->forward(state_batch, jointAction));
        auto actorLoss_i = (temperatures[i] * currentLogProbParts[i] - qMin_i).mean();

        actor_optimizers[i].zero_grad();
        actorLoss_i.backward();
        actor_optimizers[i].step();
        actor_loss_log[i] = actorLoss_i.item<double>();  // ← 新增

        auto alphaLoss_i = -(log_alphas[i] * (currentLogProbParts[i].detach() + target_entropy)).mean();
        alpha_optimizers[i].zero_grad();
        alphaLoss_i.backward();
        alpha_optimizers[i].step();
        temperatures[i] = log_alphas[i].exp().item<double>();
    }

    // --- soft update ---
    {
        torch::NoGradGuard noGrad;
        for (int i = 0; i < num_agents; ++i) {
            soft_update(critics1[i], target_critics1[i]);
            soft_update(critics2[i], target_critics2[i]);
        }
    }

    // --- log to CSV ---
    ++update_step_count;
    if (update_step_count % 50 == 0)
    {
        system("mkdir -p ./txtoutput");

        bool write_header = false;
        {
            std::ifstream check("./txtoutput/train_log.csv");
            write_header = !check.good();
        }

        std::ofstream f("./txtoutput/train_log.csv", std::ios::app);
        if (write_header)
        {
            f << "step,time";
            for (int i = 0; i < num_agents; ++i) f << ",alpha_" << i;
            for (int i = 0; i < num_agents; ++i) f << ",actor_loss_" << i;
            for (int i = 0; i < num_agents; ++i) f << ",critic_loss_" << i;
            f << "\n";
        }

        f << update_step_count << "," << time;
        for (int i = 0; i < num_agents; ++i) f << "," << temperatures[i];
        for (int i = 0; i < num_agents; ++i) f << "," << actor_loss_log[i];
        for (int i = 0; i < num_agents; ++i) f << "," << critic_loss_log[i];
        f << "\n";
    }
}


// Optionally: save/load model weights
void MultiAgents::save(const std::string& path_prefix) const {
    // Save per-agent: actor / critic1 / critic2 / target_critic1 / target_critic2 / log_alpha
    for (int i = 0; i < num_agents; ++i) {
        const std::string base = path_prefix + "_agent" + std::to_string(i);

        torch::save(actors[i],           base + "_actor.pt");
        torch::save(critics1[i],         base + "_critic1.pt");
        torch::save(critics2[i],         base + "_critic2.pt");
        torch::save(target_critics1[i],  base + "_target_critic1.pt");
        torch::save(target_critics2[i],  base + "_target_critic2.pt");
        torch::save(log_alphas[i],       base + "_log_alpha.pt");
    }

    // Save temperatures (alpha scalar values) for seamless training continuation
    torch::Tensor temps = torch::empty({num_agents}, torch::TensorOptions().dtype(torch::kFloat64));
    for (int i = 0; i < num_agents; ++i) temps[i] = temperatures[i];
    torch::save(temps, path_prefix + "_temperatures.pt");

    // (Optional) save dimension info for debugging/self-check (can be removed if not needed)
    torch::Tensor act_dims = torch::from_blob(
        const_cast<int*>(local_act_dim.data()),
        {(long)local_act_dim.size()},
        torch::TensorOptions().dtype(torch::kInt32)
    ).clone();
    torch::Tensor obs_offsets = torch::from_blob(
        const_cast<int64_t*>(local_obs_offsets.data()),
        {(long)local_obs_offsets.size()},
        torch::TensorOptions().dtype(torch::kInt64)
    ).clone();
    torch::save(act_dims,    path_prefix + "_act_dims.pt");
    torch::save(obs_offsets, path_prefix + "_obs_offsets.pt");

    std::cout << "[MultiAgents::save] checkpoint saved at prefix: " << path_prefix << std::endl;
}

void MultiAgents::load(const std::string& path_prefix) {
    // Load per-agent: actor / critic1 / critic2 / (if exists load target_*, otherwise hard_update) / log_alpha
    for (int i = 0; i < num_agents; ++i) {
        const std::string base = path_prefix + "_agent" + std::to_string(i);

        torch::load(actors[i],     base + "_actor.pt");
        torch::load(critics1[i],   base + "_critic1.pt");
        torch::load(critics2[i],   base + "_critic2.pt");
        torch::load(log_alphas[i], base + "_log_alpha.pt");

        // Try to load target critics; if not found, hard update from critics
        bool loaded_target = true;
        try {
            torch::load(target_critics1[i], base + "_target_critic1.pt");
            torch::load(target_critics2[i], base + "_target_critic2.pt");
        } catch (...) {
            loaded_target = false;
        }
        if (!loaded_target) {
            hard_update(critics1[i], target_critics1[i]);
            hard_update(critics2[i], target_critics2[i]);
        }

        // Restore temperature (alpha): derived from log_alpha
        temperatures[i] = log_alphas[i].exp().item<double>();
    }

    // Try to restore the temperature vector if available
    try {
        torch::Tensor temps;
        torch::load(temps, path_prefix + "_temperatures.pt");
        temps = temps.to(torch::kCPU);
        for (int i = 0; i < num_agents && i < temps.numel(); ++i) {
            temperatures[i] = temps[i].item<double>();
        }
    } catch (...) {
        // If no temperature file is saved, fallback values from log_alphas are used
    }

    std::cout << "[MultiAgents::load] checkpoint loaded from prefix: " << path_prefix << std::endl;
}


//- Load a single-agent checkpoint and broadcast to all agents (reviewer requested for homogeneous agent scenarios)
void MultiAgents::loadSingleModelToAll(const std::string& path_prefix) {
    const std::string base = path_prefix + "_agent0";

    // Load actor and log_alpha from single-agent checkpoint
    torch::load(actors[0],     base + "_actor.pt");
    torch::load(log_alphas[0], base + "_log_alpha.pt");
    temperatures[0] = log_alphas[0].exp().item<double>();

    // Broadcast actor weights to all remaining agents (deep copy)
    for (int i = 1; i < num_agents; ++i) {
        copy_module_params(actors[0], actors[i]);

        torch::NoGradGuard no_grad;
        log_alphas[i].data().copy_(log_alphas[0].data());
        temperatures[i] = temperatures[0];
    }

    std::cout << "[MultiAgents::loadSingleModelToAll] single-agent actor "
              << "broadcast to all " << num_agents << " agents from: "
              << path_prefix << std::endl;
}