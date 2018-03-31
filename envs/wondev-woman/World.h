#ifndef WORLD_H
#define WORLD_H

#include <vector>
#include <iostream>

#include "Player.h"
#include "Grid.h"
#include "constants.h"
#include "Action.h"
#include "ActionType.h"
#include "ActionResult.h"
#include "Direction.h"

class World {
public:

    ///////// Les fonctions qui t'intéressent

    bool gameIsOver() {
        bool oneDead = this->players[0].dead;
        bool twoDead = this->players[1].dead;

        if (oneDead && twoDead) {
            return true;
        } else if (oneDead && !twoDead) {
            return this->players[1].score > this->players[0].score;
        } else if (!oneDead && twoDead) {
            return this->players[0].score > this->players[1].score;
        } else {
            return this->round > MAX_ROUND_COUNT * 2;
        }
    }

    vector<int> getObservationForPlayer(int playerIdx) {

        // TODO : le truc pour préallouer
        vector<int> observation;

        const Player& self = this->players[playerIdx];
        const Player& other = this->players[(playerIdx + 1) % 2];

        // Attention : non donné sur CG, il faudra le maintenir dans le python
        observation.push_back(self.score);

        // Attention : non donné sur CG, il faudra fournir un résultat (incertain, probablement prendre la borne sup) dans le python
        observation.push_back(other.score);

        for (int y = 0; y < MAXIMUM_MAP_SIZE; ++y) {
            for (int x = 0; x < MAXIMUM_MAP_SIZE; ++x) {
                observation.push_back(grid.get(x, y));
            }
        }

        for (int i = 0; i < 2; i++) {
            observation.push_back(self.units[i].position.x);
            observation.push_back(self.units[i].position.y);
        }
        for (int i = 0; i < 2; i++) {
            if (self.canSee(other.units[i])) {
                observation.push_back(1);
                observation.push_back(other.units[i].position.x);
                observation.push_back(other.units[i].position.y);
            } else {
                observation.push_back(0);
                observation.push_back(-1);
                observation.push_back(-1);
            }
        }
        return observation;
    }

    int getScoreForPlayer(int playerIdx) {
        return this->players[playerIdx].score;
    }

    // Je met toujours une action valide, "stand by", même quand le joueur peut pas jouer
    // pour que l'adversaire continue sans que tu n'aie de cas particulier à gérer
    vector<Action> getLegalActions(int playerIdx) {
        Player& player = this->players[playerIdx];
        vector<Action> actions;

        for (int i = 0; i < 2; i++) {
            for (Direction dir1 : DIRECTION_VALUES) {
                for (Direction dir2 : DIRECTION_VALUES) {
                    if (this->computeAction(ActionType::MOVE, &(player.units[i]), dir1, dir2).isValid()) {
                        actions.push_back(Action(ActionType::MOVE, i, dir1, dir2));
                    }
                    if (this->computeAction(ActionType::PUSH, &(player.units[i]), dir1, dir2).isValid()) {
                        actions.push_back(Action(ActionType::PUSH, i, dir1, dir2));
                    }
                }
            }
        }

        if (actions.size() == 0) {
            actions.push_back(STAND_BY);
        }

        return actions;
    }

    void playActionForPlayer(int playerIdx, const Action& action) {

        this->round++;

        if (action.type == ActionType::STAND_BY) {
            return;
        }

        Player& player = this->players[playerIdx];

        ActionResult ar = this->computeAction(action.type, &(player.units[action.unitIndex]), action.dir1, action.dir2);
        if (ar.type == ActionType::INVALID) {
            std::cerr << "ERROR : INVALID ACTION PLAYED" << std::endl;
            player.die();
            return;
        }
        if (ar.moveValid) {
            ar.unit->position = ar.moveTarget;
        }
        if (ar.placeValid) {
            this->grid.place(ar.placeTarget);
        }
        if (ar.scorePoint) {
            player.score++;
        }
    }

    ////////////// Normalement à partir de la tu t'en fous

    int round = 0;
    Grid grid;
    Player players[2];

    World () {
        // TODO : get random position for units based on grid
    }

    ActionResult computeAction(ActionType actionType, Unit* unit, Direction dir1, Direction dir2) {
        if (actionType == ActionType::MOVE) {
            return computeMove(unit, dir1, dir2);
        } else if (actionType == ActionType::PUSH) {
            return computePush(unit, dir1, dir2);
        } else {
            return INVALID_ACTION;
        }
    }

    ActionResult computeMove(Unit* unit, Direction dir1, Direction dir2) {

        Point target = unit->position.getNeighbor(dir1);
        int targetHeight = grid.get(target);
        int currentHeight = grid.get(unit->position);

        if (targetHeight > currentHeight + 1) {
            return INVALID_ACTION;
        }
        if (targetHeight >= FINAL_HEIGHT) {
            return INVALID_ACTION;
        }
        if (this->getUnitOnPoint(target) != NULL) {
            return INVALID_ACTION;
        }

        Point placeTarget = target.getNeighbor(dir2);
        if (grid.get(placeTarget) >= FINAL_HEIGHT) {
            return INVALID_ACTION;
        }

        ActionResult result = ActionResult(ActionType::MOVE);
        result.moveTarget = target;
        result.placeTarget = placeTarget;

        Unit* possibleUnit = getUnitOnPoint(placeTarget);
        if (possibleUnit == NULL || possibleUnit == unit) {
            result.placeValid = true;
            result.moveValid = true;
        } else if (!unit->player->canSee(*possibleUnit)) {
            result.placeValid = false;
            result.moveValid = true;
        } else {
            return INVALID_ACTION;
        }

        if (targetHeight == FINAL_HEIGHT - 1) {
            result.scorePoint = true;
        }
        result.unit = unit;
        return result;
    }

    ActionResult computePush(Unit* unit, Direction dir1, Direction dir2) {
        if (!validPushDirection(dir1, dir2)) {
            return INVALID_ACTION;
        }
        Point target = unit->position.getNeighbor(dir1);
        Unit* pushed = getUnitOnPoint(target);
        if (pushed == NULL) {
            return INVALID_ACTION;
        }

        if (pushed->player == unit->player) {
            return INVALID_ACTION;
        }

        Point pushTo = pushed->position.getNeighbor(dir2);
        int toHeight = grid.get(pushTo);
        int fromHeight = grid.get(target);

        if (toHeight >= FINAL_HEIGHT || toHeight > fromHeight + 1) {
            return INVALID_ACTION;
        }

        ActionResult result = ActionResult(ActionType::PUSH);
        result.moveTarget = pushTo;
        result.placeTarget = target;

        Unit* possibleUnit = getUnitOnPoint(pushTo);
        if (possibleUnit == NULL) {
            result.placeValid = true;
            result.moveValid = true;
        } else if (!unit->player->canSee(*possibleUnit)) {
            result.placeValid = false;
            result.moveValid = false;
        } else {
            return INVALID_ACTION;
        }

        result.unit = pushed;
        return result;
    }

    Unit* getUnitOnPoint(const Point& target) {
        for (Player& p: this->players) {
            for (int i = 0; i < 2; ++i) {
                if (p.units[i].position == target) {
                    return &(p.units[i]);
                }
            }
        }
        return NULL;
    }
};

#endif
