#ifndef UNIT_H
#define UNIT_H

#include "Point.h"

class Player;

class Unit {
public:
    Player* player;
    Point position;

    Unit() {}
    Unit(Player* p): player(p) {}
};

#endif
