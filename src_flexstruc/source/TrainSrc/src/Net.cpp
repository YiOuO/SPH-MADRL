#include "TrainSrc/include/Net.h"

ActorImpl::ActorImpl(int input_dim_, int output_dim_)
{
    input_dim = input_dim_;
    fc1_hidden_size = 256;
    fc2_hidden_size = 256;
    fc3_hidden_size = 128;
    output_dim = output_dim_;

    fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(input_dim, fc1_hidden_size)));
    fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(fc1_hidden_size, fc2_hidden_size)));
    fc3 = register_module("fc3", torch::nn::Linear(torch::nn::LinearOptions(fc2_hidden_size, fc3_hidden_size)));
    // fc_output = register_module("fc_output", torch::nn::Linear(torch::nn::LinearOptions(fc3_hidden_size, output_dim)));
    mean = register_module("mean", torch::nn::Linear(fc3_hidden_size, output_dim));
    log_std = register_module("log_std", torch::nn::Linear(fc3_hidden_size, output_dim));

    this->apply(weights_init_);

}

    std::tuple<torch::Tensor, torch::Tensor> ActorImpl::forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::relu(fc3->forward(x));
        auto mu = mean->forward(x);
        auto log_std_out = torch::clamp(log_std->forward(x), -20, 2); // numerical stability
        return std::make_tuple(mu, log_std_out);
    }

    std::tuple<torch::Tensor, torch::Tensor> ActorImpl::sample(torch::Tensor x) {
        auto [mu, log_std] = forward(x);
        auto std = torch::exp(log_std);
        auto eps = torch::randn_like(std);
        auto action_raw = mu + std * eps;
        auto action = action_raw.tanh();
        auto log_prob = (-0.5 * (eps.pow(2) + 2*log_std + std::log(2*M_PI)) - 
                        torch::log(1 - action.pow(2) + 1e-6)); 
        log_prob = log_prob.sum(1, true);
        return {action, log_prob};
    }

    torch::Tensor ActorImpl::predict(torch::Tensor x)
    {
        auto [mu, log_std] = forward(x);
        auto action = mu.tanh();
        return action;
    }


CriticImpl::CriticImpl(int64_t input_dim, int64_t action_dim)
{
    fc1_hidden_size = 256;
    fc2_hidden_size = 256;
    fc3_hidden_size = 128;
    output_dim = 1;

    fc1_q = register_module("fc1_q", torch::nn::Linear(torch::nn::LinearOptions(input_dim+action_dim, fc1_hidden_size)));
    fc2_q = register_module("fc2_q", torch::nn::Linear(torch::nn::LinearOptions(fc1_hidden_size, fc2_hidden_size)));
    fc3_q = register_module("fc3_q", torch::nn::Linear(torch::nn::LinearOptions(fc2_hidden_size, fc3_hidden_size)));
    fc_output_q = register_module("fc_output_q", torch::nn::Linear(torch::nn::LinearOptions(fc3_hidden_size, output_dim)));

    this->apply(weights_init_);
}

torch::Tensor CriticImpl::forward(torch::Tensor state, torch::Tensor action) {
    auto x = torch::cat({state, action}, 1);
    x = torch::relu(fc1_q->forward(x));
    x = torch::relu(fc2_q->forward(x));
    x = torch::relu(fc3_q->forward(x));
    return fc_output_q->forward(x);
}


void weights_init_(torch::nn::Module& module) {
    if (auto* linear = module.as<torch::nn::Linear>()) {
        torch::nn::init::kaiming_uniform_(linear->weight, std::sqrt(5));
        if (linear->bias.defined()) {
            auto fan_in = linear->weight.size(1);
            auto bound = 1.0 / std::sqrt(fan_in);
            torch::nn::init::uniform_(linear->bias, -bound, bound);
        }
    }
}