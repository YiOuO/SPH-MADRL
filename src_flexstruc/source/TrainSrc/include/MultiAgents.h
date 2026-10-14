#ifndef MULTIAGENTS_H_
#define MULTIAGENTS_H_

#include <iostream>
#include <torch/torch.h>
#include "TrainSrc/include/Net.h"
#include <filesystem>
#include <fstream>

class MultiAgents
{
private:
    // Multi agents parameters
    int num_agents;
    std::vector<int> local_obs_dim;
    std::vector<int> local_act_dim;
    int global_state_dim;
    int joint_act_dim;
    // NN param
    double lr;
    double alpha;
    double alpha_lr;
    double target_entropy;
    double tau;
    double gamma;
    int policy_freq;
    int total_iteration;
    torch::Device my_device;


    std::vector<int64_t> local_obs_offsets;
    int update_step_count = 0;

public:
    // Per-agent policies and temperatures
    std::vector<Actor> actors;
    std::vector<torch::optim::Adam> actor_optimizers;

    std::vector<torch::Tensor> log_alphas;
    std::vector<double> temperatures;
    std::vector<torch::optim::Adam> alpha_optimizers;

    // Per-agent centralized twin critics + targets
    std::vector<Critic> critics1, critics2, target_critics1, target_critics2;
    std::vector<std::unique_ptr<torch::optim::Adam>> critic1_optimizers, critic2_optimizers;

public:
    MultiAgents(
        int numberOfAgents,
        std::vector<int> localObservationDimension,
        std::vector<int> actionDimension,
        float initialAlpha,
        float alphaLearningRate,
        float discountFactor,
        float targetUpdateTau,
        float learningRate,
        int policyFreq,
        bool use_cuda)
        : my_device(use_cuda && torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU)),
        num_agents(numberOfAgents),
        local_obs_dim(std::move(localObservationDimension)),
        local_act_dim(std::move(actionDimension)),
        global_state_dim(std::accumulate(local_obs_dim.begin(), local_obs_dim.end(), 0)),
        joint_act_dim(std::accumulate(local_act_dim.begin(), local_act_dim.end(), 0)),
        lr(learningRate),
        tau(targetUpdateTau),
        policy_freq(policyFreq),
        alpha(initialAlpha),
        alpha_lr(alphaLearningRate),
        gamma(discountFactor),
        target_entropy(-static_cast<float>(
            std::accumulate(local_act_dim.begin(), local_act_dim.end(), 0.0)
        )),
        total_iteration(0)
    {
        // Per-agent actor and temperature
        actors.reserve(num_agents);
        actor_optimizers.reserve(num_agents);
        log_alphas.reserve(num_agents);
        alpha_optimizers.reserve(num_agents);
        temperatures.resize(num_agents);

        for (int i = 0; i < num_agents; ++i)
        {
            actors.emplace_back(Actor(local_obs_dim[i], local_act_dim[i]));
            actors.back()->to(my_device);
            actors.back()->to(torch::kF32);
            actor_optimizers.emplace_back(actors.back()->parameters(), lr);

            auto logAlpha = torch::log(torch::tensor({alpha}, torch::TensorOptions().dtype(torch::kF32).device(my_device))).set_requires_grad(true);
            
            log_alphas.push_back(logAlpha);
            alpha_optimizers.emplace_back(std::vector<torch::Tensor>{log_alphas.back()}, alpha_lr);
            temperatures[i] = alpha;
        }
        
        // Per-agent critics and targets
        critics1.reserve(num_agents);
        critics2.reserve(num_agents);
        target_critics1.reserve(num_agents);
        target_critics2.reserve(num_agents);
        critic1_optimizers.reserve(num_agents);
        critic2_optimizers.reserve(num_agents);

        for (int i = 0; i < num_agents; ++i)
        {
            critics1.emplace_back(Critic(global_state_dim, joint_act_dim));
            critics2.emplace_back(Critic(global_state_dim, joint_act_dim));
            target_critics1.emplace_back(Critic(global_state_dim, joint_act_dim));
            target_critics2.emplace_back(Critic(global_state_dim, joint_act_dim));

            critics1.back()->to(my_device);
            critics1.back()->to(torch::kF32);
            critics2.back()->to(my_device);
            critics2.back()->to(torch::kF32);

            target_critics1.back()->to(my_device);
            target_critics1.back()->to(torch::kF32);
            target_critics2.back()->to(my_device);
            target_critics2.back()->to(torch::kF32);

            hard_update(critics1.back(), target_critics1.back());
            hard_update(critics2.back(), target_critics2.back());

            critic1_optimizers.emplace_back(std::make_unique<torch::optim::Adam>(critics1.back()->parameters(), lr));
            critic2_optimizers.emplace_back(std::make_unique<torch::optim::Adam>(critics2.back()->parameters(), lr));
        }


        // todo test different states of agents
        local_obs_offsets.resize(local_obs_dim.size() + 1);
        local_obs_offsets[0] = 0;
        for (size_t i = 0; i < local_obs_dim.size(); ++i) {
            local_obs_offsets[i + 1] = local_obs_offsets[i] + local_obs_dim[i];
        }        
    }

    
public:

    // Forward a batch global state to extract the local slice for agent i
    torch::Tensor sliceLocalObservation(const torch::Tensor& batchGlobalState, int agentIndex);
    
    // Concatenate along feature dimension
    torch::Tensor concatAlongFeature(const std::vector<torch::Tensor> &parts);


    void update(    const torch::Tensor& state_batch_, 
                    const torch::Tensor& action_batch_, 
                    const torch::Tensor& next_state_batch_, 
                    const torch::Tensor& done_batch_,
                    const torch::Tensor& reward_batch_
                );

    void update(    const torch::Tensor& state_batch_, 
                    const torch::Tensor& action_batch_, 
                    const torch::Tensor& next_state_batch_, 
                    const torch::Tensor& done_batch_,
                    const torch::Tensor& reward_batch_,
                    const float time
                );

    void save(const std::string &saveModelDir) const;

    void load(const std::string &loadModelDir);
    void loadSingleModelToAll(const std::string &loadModelDir); //-Single model to all agents (for homogeneous agents)

    void setNetMode(bool trainOrEval);

    std::vector<torch::Tensor> selectActionsWithNoise(const std::vector<std::vector<float>>& inputs);
    std::vector<torch::Tensor> selectActionsWithoutNoise(const std::vector<std::vector<float>>& inputs);
    std::vector<torch::Tensor> selectRandomActions();

    // Soft update: target ← (1−τ)·target + τ·source
    void soft_update(const Critic &source, Critic &target)
    {
        torch::NoGradGuard no_grad;
        auto target_params = target->parameters();
        auto source_params = source->parameters();
        for (size_t i = 0; i < target_params.size(); ++i)
        {
            target_params[i].data().mul_(1.0 - tau);
            target_params[i].data().add_(tau * source_params[i].data());
        }
    }

    // Hard update: target ← source
    void hard_update(const Critic &source, Critic &target)
    {
        torch::NoGradGuard no_grad;
        auto target_params = target->parameters();
        auto source_params = source->parameters();
        for (size_t i = 0; i < target_params.size(); ++i)
        {
            target_params[i].data().copy_(source_params[i].data());
        }
    }


    // Deep-copy all parameters and buffers from src to dst (assumes identical architecture)
    template <typename ModuleType>
    void copy_module_params(const ModuleType& src, ModuleType& dst) {
        torch::NoGradGuard no_grad;

        auto src_params = src->parameters();
        auto dst_params = dst->parameters();
        for (size_t j = 0; j < src_params.size(); ++j) {
            dst_params[j].data().copy_(src_params[j].data());
        }

        // Copy buffers as well (e.g. BatchNorm running stats)
        auto src_bufs = src->buffers();
        auto dst_bufs = dst->buffers();
        for (size_t j = 0; j < src_bufs.size(); ++j) {
            dst_bufs[j].data().copy_(src_bufs[j].data());
        }
    }
};

#endif