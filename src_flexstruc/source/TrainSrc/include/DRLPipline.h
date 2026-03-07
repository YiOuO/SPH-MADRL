#ifndef DRLPIPLINE_H_
#define DRLPIPLINE_H_

#include "TrainSrc/include/Agent.h"
#include "TrainSrc/include/MultiAgents.h"
#include "TrainSrc/include/DRLParam.h"
#include "TrainSrc/include/ReplayBuffer.h"
#include "TypesDef.h"
#include <torch/utils.h>

class DRLPipline{
private:
    std::shared_ptr<MyAgent> my_agent;
    std::shared_ptr<MultiAgents> my_multi_agents;
    // std::shared_ptr<Environment> my_env;
    std::shared_ptr<ReplayBuffer> my_replay_buffer;
    std::shared_ptr<DRLParam> my_param;
    std::vector<float> reward_total;
    int num_agents;

public:    
    DRLPipline(int num_agents, std::vector<int> local_obs_dim, std::vector<int> local_act_dim, 
    int policy_freq, int buffer_size, float alpha, float alpha_lr, 
    float lr, float tau, float gamma, bool use_cuda);

    ~DRLPipline();

    std::vector<float> getTotalReward();
    void writeTotalRewardToFile(std::string reward_filename, bool first_flag);
    float getAction(const std::vector<float>& nn_input);
    std::vector<float> getAction(const std::vector<float>& nn_input, bool training);
    std::vector<std::vector<float>> getMultiActions(const std::vector<std::vector<float>>& nn_inputs, bool training, bool run_mode);
    void saveMultiModels();
    void loadMultiModels();
    void broadcastSingleModel();
    void setNetMode(bool trainOrEval);
    int getReplayBufferDataNum();
    void updateNet( const torch::Tensor& state_batch, 
                    const torch::Tensor& action_batch, 
                    const torch::Tensor& next_state_batch, 
                    const torch::Tensor& done_batch, 
                    const torch::Tensor& reward_batch);
    void sampleFromReplayBuffer(int batch_num, 
                                torch::Tensor& state_batch, 
                                torch::Tensor& action_batch, 
                                torch::Tensor& next_state_batch, 
                                torch::Tensor& done_batch, 
                                torch::Tensor& reward_batch);
    void storeToReplayBuffer(   const std::vector<float>& state,
                                const std::vector<float>& action,
                                const std::vector<float>& next_state,
                                const int& done,
                                const float& reward);

    void storeToReplayBuffer(   const std::vector<std::vector<float>>& state,
                                const std::vector<std::vector<float>>& action,
                                const std::vector<std::vector<float>>& next_state,
                                const std::vector<int>& done,
                                const std::vector<float>& reward);
    void updateNet(int batch_size);
    void updateMultiNet(int batch_size);
    void updateMultiNet(int batch_size, float time);

    std::vector<float> getMultiRewardAcplate(const int num_agents,const int steps_train
    ,const std::vector<float> & plate_dz, const std::vector<float>& sum_wgreward);
    
    std::vector<float> getMultiRewardWecPto(int num_agents,const std::vector<float> & sum_stepreward);
    
    std::vector<float> getMultiRewardWecPto(int num_agents, const std::vector<float>& sum_stepreward,const std::vector<float> &last_pto, const std::vector<float> &value_pto);
};

inline void DRLPipline::setNetMode(bool trainOrEval)
{
    my_agent->setNetMode(trainOrEval);
}

inline int DRLPipline::getReplayBufferDataNum()
{
    return my_replay_buffer->getBufferDataNum();
}

inline std::vector<float> DRLPipline::getTotalReward()
{
    return reward_total;
}

#endif