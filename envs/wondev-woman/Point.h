#ifndef POINT_H
#define POINT_H

#include "Direction.h"

class Point {
public:
    int x, y;

    Point () {}

    Point(int x, int y) {
        this->x = x;
        this->y = y;
    }

    bool operator==(const Point& p) {
        return this->x == p.x && this->y == p.y;
    }

    int distance(Point other) const {
        return std::max(std::abs(x - other.x), std::abs(y - other.y));
    }

    int toHash() const {
        return this->x * 50 + this->y;
    }

    Point getNeighbor (Direction d) const {
        return Point(
            x + (d == E) + (d == NE) + (d == SE) - (d == W) - (d == NW) - (d == SW),
            y + (d == S) + (d == SW) + (d == SE) - (d == N) - (d == NW) - (d == NE)
        );
    }

};

#endif
