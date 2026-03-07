#ifndef ENV_H_
#define ENV_H_

#include <iostream>
#include <torch/torch.h>

class Environment{
private:

public:
    Environment();
    ~Environment();

    /*
    -----------------------------------
    功能：在仿真环境中执行一步操作
    @param state        当前波浪状态
    @param action       当前执行的动作
    @param reward       返回当前动作的奖励值
    @param next_state   返回执行该动作后的波浪状态
    -----------------------------------
    */
    void step();

    /*
    -----------------------------------
    功能：计算奖励值
    @param state        待评价的波浪状态

    @return             奖励值
    -----------------------------------
    */
    double calReward();

    /*
    -----------------------------------
    功能：重置环境
    -----------------------------------
    */
    void reset();
};

#endif