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
// #include <torch/torch.h>
// #include "TrainSrc/include/Net.h"

// class MyAgent{
// private:
//     double learning_rate;
//     double max_action;
//     double discount;
//     double tau;
//     double policy_noise;
//     double noise_clip;
//     int policy_freq;
//     int total_iteration;
//     int state_dim;
//     int action_dim;
//     double baseaction;

//     torch::Device my_device;

//     std::shared_ptr<ActorNet> actor;
//     std::shared_ptr<ActorNet> actor_target;
//     std::shared_ptr<torch::optim::Adam> actor_optimizer;

//     std::shared_ptr<CriticNet> critic;
//     std::shared_ptr<CriticNet> critic_target;
//     std::shared_ptr<torch::optim::Adam> critic_optimizer;


// public:
//     MyAgent(int state_dim_,
//             int action_dim_,
//             double max_action_,
//             double discount_,
//             double tau_,
//             double policy_noise_,
//             int policy_freq_,
//             double learning_rate_,
//             double noise_clip_,
//             bool use_cuda,
//             double baseaction_
//     );
//     ~MyAgent();

//     /**
//      * 
//      * 功能：TD3算法的实现，训练网络
//      * @param state_batch_         一批当前波浪状态
//      * @param action_batch_        一批当前波浪状态对应的波浪板动作
//      * @param next_state_batch_    一批执行动作后对应的波浪状态
//      * @param reward_batch_        一批执行动作后对应的奖励
//      * @param done_batch_          执行动作后是否已结束
//      * 
//     */
//     void update(    const torch::Tensor& state_batch_, 
//                     const torch::Tensor& action_batch_, 
//                     const torch::Tensor& next_state_batch_, 
//                     const torch::Tensor& done_batch_,
//                     const torch::Tensor& reward_batch_
//                 );
//     /**
//      * 
//      * 功能：保存网络结构参数
//      * @param saveModelDir         保存的地址
//      *
//     */
//     void save(const std::string& saveModelDir);

//     /**
//      *
//      * 功能：加载网络结构参数
//      * @param loadModelDir         读取的地址
//      *
//     */
//     void load(const std::string& loadModelDir);

//     /**
//      *
//      * 功能：设置网络的模式为训练模式还是评估模式
//      * @param trainOrEval          true为训练模式；false为评估模式
//      *
//     */
//     void setNetMode(bool trainOrEval);

//     /**
//      * 
//      * 功能：将当前状态输入网络，得到一个动作
//      * @param input                当前波浪状态
//      * 
//     */
//     torch::Tensor selectAction(std::vector<float>& input);

//     torch::Tensor selectActionWithNoise(std::vector<float>& input);

//     torch::Tensor selectRandomAction();


//     torch::Tensor selectActionBaseaction(std::vector<float>& input);
//     torch::Tensor selectActionWithNoiseBaseaction(std::vector<float>& input);
//     torch::Tensor selectRandomActionBaseaction();
// };

// inline void MyAgent::setNetMode(bool trainOrEval)
// {
//     // if true, then train mode; if false, then eval mode
//     trainOrEval ? actor->train() : actor->eval();
// }

// #endif