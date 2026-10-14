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
    Execute one simulation step.
    @param state        Current wave state.
    @param action       Action to execute.
    @param reward       Reward for the action.
    @param next_state   Wave state after the action.
    -----------------------------------
    */
    void step();

    /*
    -----------------------------------
    Compute the reward for a wave state.
    @param state        Wave state to evaluate.

    @return             Computed reward.
    -----------------------------------
    */
    double calReward();

    /*
    -----------------------------------
    Reset the environment.
    -----------------------------------
    */
    void reset();
};

#endif
