#include "TrainSrc/include/Agent.h"

MyAgent::MyAgent(   int state_dim_,
                    int action_dim_,
                    double alpha_,
                    double alpha_lr_,
                    double gamma_,
                    double tau_,
                    int policy_freq_,
                    double lr_,
                    bool use_cuda
                ):
      my_device(use_cuda && torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU)),
      lr(lr_),
      gamma(gamma_),
      tau(tau_),
      policy_freq(policy_freq_),
      state_dim(state_dim_),
      action_dim(action_dim_),
      alpha(alpha_),
      alpha_lr(alpha_lr_),
      total_iteration(0),
      target_entropy(-static_cast<double>(action_dim_)),
      actor(state_dim_, action_dim_),
      critic1(state_dim_ , action_dim_),
      critic2(state_dim_ , action_dim_),
      target1(state_dim_ , action_dim_),
      target2(state_dim_ , action_dim_),
      log_alpha(torch::log(torch::tensor({alpha_}, torch::kF32)).set_requires_grad(true)), 
      alpha_opt({log_alpha}, alpha_lr_), 
      actor_opt(actor->parameters(), lr_),
      critic1_opt(critic1->parameters(), lr_),
      critic2_opt(critic2->parameters(), lr_)
      {
        hard_update(critic1, target1);
        hard_update(critic2, target2);

        actor  ->to(torch::kF32);
        critic1->to(torch::kF32);
        critic2->to(torch::kF32);
        target1->to(torch::kF32);
        target2->to(torch::kF32);

      }

    torch::Tensor MyAgent::selectActionWithNoise(std::vector<float>& input)
    {
        torch::NoGradGuard no_grad;
        unsigned int input_size = input.size();
        torch::Tensor state = torch::from_blob(input.data(), {1, input_size}, torch::TensorOptions().dtype(torch::kFloat32)).clone().to(my_device);
        auto [action, log_prob] = actor->sample(state);
        std::cout<< "noise action "<< action[0][0].item().toFloat()<<std::endl;
        return action.cpu().clone();
    }

    torch::Tensor MyAgent::selectRandomAction()
    {
        torch::Tensor output_tmp = torch::zeros({1, 1},torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor noise = (torch::rand_like(output_tmp) - 0.5) * 2; //[-1,1]
        std::cout<< "random action "<< noise[0][0].item().toFloat()<<std::endl;
        return noise.cpu().clone();
    }

    torch::Tensor MyAgent::selectAction(std::vector<float>& input)
    {   
        torch::NoGradGuard no_grad;
        unsigned int input_size = input.size();
        torch::Tensor state = torch::from_blob(input.data(), {1, input_size}, torch::TensorOptions().dtype(torch::kFloat32)).clone().to(my_device);
        auto [action, log_prob] = actor->forward(state);
        std::cout << "net_output: " << action[0][0].item().toFloat() << std::endl;
        return action.cpu().clone();
    }
      
    /**
     * Performs one training step using a mini-batch from the replay buffer.
     * If there are not enough samples, the function returns without doing anything.
     */
    // void train_step(ReplayBuffer& buffer, size_t batch_size)

    void MyAgent::update(   const torch::Tensor& state_batch_, 
                            const torch::Tensor& action_batch_, 
                            const torch::Tensor& next_state_batch_, 
                            const torch::Tensor& done_batch_,
                            const torch::Tensor& reward_batch_
                        )
    {
        //---------- Critic update ----------
        torch::Tensor q_target;
        {
            torch::NoGradGuard no_grad;

            auto [a_next, logp_next] = actor->sample(next_state_batch_);
            logp_next = logp_next.detach();
            auto q1_next = target1->forward(next_state_batch_, a_next);
            auto q2_next = target2->forward(next_state_batch_, a_next);
            q_target = reward_batch_ +
                    gamma * (1 - done_batch_) *
                    (torch::min(q1_next, q2_next) - alpha * logp_next);
        }       
        auto q1 = critic1->forward(state_batch_, action_batch_);
        auto q2 = critic2->forward(state_batch_, action_batch_);
        auto critic_loss = torch::mse_loss(q1, q_target) + torch::mse_loss(q2, q_target);

        critic1_opt.zero_grad();
        critic2_opt.zero_grad();
        critic_loss.backward();
        critic1_opt.step();
        critic2_opt.step();

        //---------- Actor update ----------
        auto [a_pred, logp_pred] = actor->sample(state_batch_);
        auto q_pred = torch::min(critic1->forward(state_batch_, a_pred),
                                 critic2->forward(state_batch_, a_pred));
        auto actor_loss = (alpha * logp_pred - q_pred).mean();

        actor_opt.zero_grad();
        actor_loss.backward();
        actor_opt.step();

        //---------- Alpha update ----------
        // auto alpha_loss = -(log_alpha.exp() * (logp_pred + target_entropy).detach()).mean();
        auto detached_logp = logp_pred.detach();
        auto alpha_loss = -(log_alpha * (detached_logp + target_entropy)).mean();
        alpha_opt.zero_grad();
        alpha_loss.backward();
        alpha_opt.step();
        alpha = log_alpha.exp().item<double>();  // update scalar        

        //---------- Soft update target networks ----------
        // torch::NoGradGuard no_grad;
        soft_update(critic1, target1);
        soft_update(critic2, target2);
        // std::cout << "actor_loss: " << actor_loss.item<double>()
        //   << ", critic_loss: " << critic_loss.item<double>()
        //   << ", avg_q: " << q_pred.mean().item<double>()
        //   << ", logp: " << logp_pred.mean().item<double>() << std::endl;
    }

    // Optionally: save/load model weights
    void MyAgent::save(const std::string& path_prefix) const {
        torch::save(actor,   path_prefix + "_actor.pt");
        torch::save(critic1, path_prefix + "_critic1.pt");
        torch::save(critic2, path_prefix + "_critic2.pt");
        torch::save(log_alpha, path_prefix + "_log_alpha.pt");
        std::cout<< "best model saved "<<std::endl;
    }

    void MyAgent::load(const std::string& path_prefix) {
        torch::load(actor,   path_prefix + "_actor.pt");
        torch::load(critic1, path_prefix + "_critic1.pt");
        torch::load(critic2, path_prefix + "_critic2.pt");
        torch::load(log_alpha, path_prefix + "_log_alpha.pt");
        hard_update(critic1, target1);
        hard_update(critic2, target2);
        alpha = log_alpha.exp().item<double>();
        std::cout<< "best model loaded "<<std::endl;
    }

    MyAgent::~MyAgent()
    {
        std::cout << "MyAgent resources cleaned up." << std::endl;
    }