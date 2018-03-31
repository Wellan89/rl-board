#ifndef ACTION_H
#define ACTION_H

#include "Direction.h"
#include "ActionType.h"

class Action {
public:
    ActionType type;
    int unitIndex;
    Direction dir1;
    Direction dir2;

    Action(ActionType t, int ui, Direction d1, Direction d2): type(t), unitIndex(ui), dir1(d1), dir2(d2) {}
};

const Action STAND_BY = Action(ActionType::STAND_BY, -1, W, W);

#endif
