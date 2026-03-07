#include "TrainSrc/include/ReplayBuffer.h"
#include <random>

ReplayBuffer::ReplayBuffer(int buffer_size_, int state_dim_, int action_dim_
,int next_state_dim_, int done_dim_, int reward_dim_)
{
    buffer_size = buffer_size_;

    state_list.reserve(buffer_size + 1);
    action_list.reserve(buffer_size + 1);
    next_state_list.reserve(buffer_size + 1);
    done_list.reserve(buffer_size + 1);
    reward_list.reserve(buffer_size + 1);

    state_dim = state_dim_;
    action_dim = action_dim_;
    next_state_dim = next_state_dim_;
    done_dim = done_dim_;
    reward_dim = reward_dim_;
}

ReplayBuffer::~ReplayBuffer()
{

}

void ReplayBuffer::store(   torch::Tensor state,
                            torch::Tensor action,
                            torch::Tensor next_state,
                            torch::Tensor done,
                            torch::Tensor reward)
{
    if(state_list.size() == buffer_size){
        state_list.erase(state_list.begin());
        action_list.erase(action_list.begin());
        next_state_list.erase(next_state_list.begin());
        done_list.erase(done_list.begin());
        reward_list.erase(reward_list.begin());
    }

    state_list.emplace_back(state);
    action_list.emplace_back(action);
    next_state_list.emplace_back(next_state);
    done_list.emplace_back(done);
    reward_list.emplace_back(reward);
}

void ReplayBuffer::sample(  int batch_num, 
                            torch::Tensor& state_batch, 
                            torch::Tensor& action_batch, 
                            torch::Tensor& next_state_batch, 
                            torch::Tensor& done_batch, 
                            torch::Tensor& reward_batch
                        )
{
    // random generator
    // std::random_device rd;
    // std::mt19937 gen;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, state_list.size() - 1);

    // init batch 
    state_batch = torch::zeros(     {batch_num, state_dim},         torch::TensorOptions().dtype(torch::kFloat32));
    action_batch = torch::zeros(    {batch_num, action_dim},        torch::TensorOptions().dtype(torch::kFloat32));
    next_state_batch = torch::zeros({batch_num, next_state_dim},    torch::TensorOptions().dtype(torch::kFloat32));
    done_batch = torch::zeros(      {batch_num, done_dim},          torch::TensorOptions().dtype(torch::kFloat32));
    reward_batch = torch::zeros(    {batch_num, reward_dim},        torch::TensorOptions().dtype(torch::kFloat32));

    // get batch
    for(int i = 0; i < batch_num; i++){

        int random_index = dis(gen);
        // equals to xxx_batch[i, 0:xxx_dim] = xxx_list[random_index]
        state_batch.index_put_(     {i, torch::indexing::Slice(0, state_dim)},      state_list[random_index]);
        action_batch.index_put_(    {i, torch::indexing::Slice(0, action_dim)},     action_list[random_index]);
        next_state_batch.index_put_({i, torch::indexing::Slice(0, next_state_dim)}, next_state_list[random_index]);
        done_batch.index_put_(      {i, torch::indexing::Slice(0, done_dim)},       done_list[random_index]);
        reward_batch.index_put_(    {i, torch::indexing::Slice(0, reward_dim)},     reward_list[random_index]);
    }
}