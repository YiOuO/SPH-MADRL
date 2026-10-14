#ifndef AGENT_H_
#define AGENT_H_

#include <iostream>
#include <torch/torch.h>
#include "TrainSrc/include/Net.h"

class MyAgent{
private:
    int state_dim;
    int action_dim;
    double lr;
    double alpha;
    double alpha_lr;
    double tau;
    double gamma;
    double target_entropy;
    int policy_freq;
    int total_iteration;
    torch::Device my_device;

// private:
//     double alpha_, gamma_, tau_, target_entropy;

public:
    Actor actor;
    Critic critic1, critic2;   // Main Q-networks
    Critic target1, target2;   // Target Q-networks
    
    torch::Tensor log_alpha;
    torch::optim::Adam actor_opt;
    torch::optim::Adam critic1_opt;
    torch::optim::Adam critic2_opt;
    torch::optim::Adam alpha_opt;

public:
    MyAgent(int state_dim_,
            int action_dim_,
            double alpha_,
            double alpha_lr_,
            double gamma_,
            double tau_,
            int policy_freq_,
            double lr_,
            bool use_cuda
    );
    ~MyAgent();

    void update(    const torch::Tensor& state_batch_, 
                    const torch::Tensor& action_batch_, 
                    const torch::Tensor& next_state_batch_, 
                    const torch::Tensor& done_batch_,
                    const torch::Tensor& reward_batch_
                );

    void save(const std::string& saveModelDir)const;

    void load(const std::string& loadModelDir);

    void setNetMode(bool trainOrEval);

    torch::Tensor selectAction(std::vector<float>& input);

    torch::Tensor selectActionWithNoise(std::vector<float>& input);

    torch::Tensor selectRandomAction();
     
    // Soft update: target ← (1−τ)·target + τ·source
    void soft_update(const Critic& source, Critic& target) {
        torch::NoGradGuard no_grad;
        auto target_params = target->parameters();
        auto source_params = source->parameters();
        for (size_t i = 0; i < target_params.size(); ++i) {
            target_params[i].data().mul_(1.0 - tau);
            target_params[i].data().add_(tau * source_params[i].data());
        }
    }

    // Hard update: target ← source
    void hard_update(const Critic& source, Critic& target) {
        torch::NoGradGuard no_grad;
        auto target_params = target->parameters();
        auto source_params = source->parameters();
        for (size_t i = 0; i < target_params.size(); ++i) {
            target_params[i].data().copy_(source_params[i].data());
        }
    }
    
};

inline void MyAgent::setNetMode(bool trainOrEval)
{
    // if true, then train mode; if false, then eval mode
    trainOrEval ? actor->train() : actor->eval();
}

#endif








// #ifndef AGENT_H_
// #define AGENT_H_

// #include <iostream>
// #endif