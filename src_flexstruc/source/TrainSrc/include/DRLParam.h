#ifndef DRLPARAM_H_
#define DRLPARAM_H_

#include<iostream>

struct DRLParam{

    std::string save_model_dir = "./txtoutput/model/"; // Directory to save the model
    std::string load_model_dir = "./txtoutput/model/"; // Directory to load the model

    // int buffer_size = 100000;         // Size of the replay buffer

    // int state_dim = 11;               // Dimension of the state
    // int action_dim = 1;               // Dimension of the action
    double max_action = 1;            // Maximum action value (e.g., max velocity)
    // double discount = 0.99;           // Discount factor (used in TD3 algorithm)
    // double tau = 0.005;               // Parameter for updating the target network
    double policy_noise = 0.02;       // Policy noise
    int policy_freq = 2;              // Frequency to update the target network
    // double learning_rate = 0.0005;     // Learning rate
    double noise_clip = 0.5;          // Noise clipping threshold
    // bool use_cuda = torch::cuda::is_available() ? true:false;
    bool use_cuda = false;
   
    // double alpha = 0.1, alpha_learning_rate = 0.001; // Entropy coefficient alpha in SAC (controls exploration);
    // int max_timestep = 100;           // Maximum number of steps per episode

    double baseaction = 400;

};

#endif