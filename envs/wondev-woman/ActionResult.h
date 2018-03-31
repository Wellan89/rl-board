#ifndef ACTION_RESULT_H
#define ACTION_RESULT_H

#include "Point.h"
#include "ActionType.h"

class ActionResult {
public:
    Point moveTarget;
    Point placeTarget;
    bool placeValid;
    bool moveValid;
    bool scorePoint;
    Unit* unit;
    ActionType type;

    bool isValid() const {
        return this->type != ActionType::INVALID;
    }

    ActionResult(ActionType t): type(t) {}
};

const ActionResult INVALID_ACTION = ActionResult(ActionType::INVALID);

#endif
