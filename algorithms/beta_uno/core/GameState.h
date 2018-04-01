#ifndef GAME_STATE_H
#define GAME_STATE_H

#include <vector>
#include <string>

class Action {};

class GameState {

    // Should deep copy
    GameState(const GameState& other);
    GameState& operator=(const GameState& other);

    bool isGameOver() const;

    int nextPlayerId() const;
    void applyAction(const Action& action);

    void computePossileActions(std::vector<Action>& actions) const;
    bool getRandomAction(Action& action) const;

    // evaluate this state and return a vector of rewards (for each agent)
    const std::vector<float> evaluate() const;

};

#endif
