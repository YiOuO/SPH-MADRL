#ifndef REPLAYBUFFER_H_
#define REPLAYBUFFER_H_

#include <iostream>
#include <vector>
#include <torch/torch.h>
#include "torch/script.h"

class ReplayBuffer{
private:
    int buffer_size;

    std::vector<torch::Tensor> state_list;
    std::vector<torch::Tensor> action_list;
    std::vector<torch::Tensor> next_state_list;
    std::vector<torch::Tensor> done_list;
    std::vector<torch::Tensor> reward_list;

    int state_dim;
    int action_dim;
    int next_state_dim;
    int done_dim;
    int reward_dim;

public:
    ReplayBuffer(int buffer_size_, int state_dim_, int action_dim_, int next_state_dim_, int done_dim_, int reward_dim_);
    ~ReplayBuffer();

    /*
    -----------------------------------
    功能：存储经验
    @param state         
    @param action         
    @param next_state         
    @param done         
    @param reward         
    -----------------------------------
    */
    void store( torch::Tensor state,
                torch::Tensor action,
                torch::Tensor next_state,
                torch::Tensor done,
                torch::Tensor reward);

    /*
    -----------------------------------
    功能：从经验池中读取batch_num个经验用于训练
    @param state_batch         
    @param action_batch         
    @param next_state_batch         
    @param done_batch         
    @param reward_batch         
    -----------------------------------
    */
    void sample(int batch_num, 
                torch::Tensor& state_batch, 
                torch::Tensor& action_batch, 
                torch::Tensor& next_state_batch, 
                torch::Tensor& done_batch, 
                torch::Tensor& reward_batch
                );
    
    /*
    -----------------------------------
    功能：获取当前经验池的大小
    @return         当前经验池的大小          
    -----------------------------------
    */
    int getBufferDataNum();
};

inline int ReplayBuffer::getBufferDataNum()
{
    return state_list.size();
}


#endif