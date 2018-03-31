#ifndef DIRECTION_H
#define DIRECTION_H

#include <vector>

enum Direction {
    NW = 0,
    N = 1,
    NE = 2,
    E = 3,
    SE = 4,
    S = 5,
    SW = 6,
    W = 7
};

const std::vector<Direction> DIRECTION_VALUES = {
    NW,
    N,
    NE,
    W,
    E,
    SW,
    S,
    SE
};

bool validPushDirection(Direction dir1, Direction dir2) {
    return (abs(dir1 - dir2) <= 1 || abs(dir1 - dir2) == 7);
}

#endif
