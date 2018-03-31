#ifndef PLAYER_H
#define PLAYER_H

#include "Unit.h"
#include "constants.h"

class Player {
public:
    int score = 0;
    bool dead, won;
    Unit units[2];

    Player() {
        this->units[0] = Unit(this);
        this->units[1] = Unit(this);
    }

    void win() {
        won = true;
    }

    void die() {
        dead = true;
    }

    bool canSee(const Unit& other) const {
        for (const Unit& u : this->units) {
            if (u.position.distance(other.position) <= VIEW_DISTANCE) {
                return true;
            }
        }
        return false;
    }
};

#endif
