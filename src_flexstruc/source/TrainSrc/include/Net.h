#ifndef NET_H_
#define NET_H_

#include <torch/torch.h>
#include <math.h>

void weights_init_(torch::nn::Module& module);
// ========== Actor Network========= //
struct ActorImpl: public torch::nn::Module{
private:
    int input_dim;
    int fc1_hidden_size;
    int fc2_hidden_size;
    int fc3_hidden_size;
    int output_dim;
    double max_action;

    torch::nn::Linear fc1 = nullptr;
    torch::nn::Linear fc2 = nullptr;
    torch::nn::Linear fc3 = nullptr;
    torch::nn::Linear fc_output = nullptr;
    torch::nn::Linear mean = nullptr;
    torch::nn::Linear log_std = nullptr;


public:
    ActorImpl(int input_dim_, int output_dim_);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input); 
    std::tuple<torch::Tensor, torch::Tensor> sample(torch::Tensor x);
    torch::Tensor predict(torch::Tensor x);
};
TORCH_MODULE(Actor);

struct CriticImpl: public torch::nn::Module{
private:
    int input_dim, output_dim;
    int fc1_hidden_size;
    int fc2_hidden_size;
    int fc3_hidden_size;

    torch::nn::Linear fc1_q = nullptr;
    torch::nn::Linear fc2_q = nullptr;
    torch::nn::Linear fc3_q = nullptr;
    torch::nn::Linear fc_output_q = nullptr;


public:
    CriticImpl(int64_t input_dim, int64_t action_dim);
    torch::Tensor forward(torch::Tensor state, torch::Tensor action);
};
TORCH_MODULE(Critic);

#endif