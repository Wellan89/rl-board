#ifndef CORE_H
#define CORE_H

#include "Node.h"
#include <cfloat>

template <class State, typename Action>
class BetaUno {
private:
    typedef Node<State, Action> TreeNode;

public:
    int iterations = 0;
    float UCT_K = sqrt(2);

    BetaUno() {}

    TreeNode* getBestUctChild(TreeNode* node) const {
        // sanity check
        if(!node->isFullyExpanded()) return NULL;

        float bestUctScore = -std::numeric_limits<float>::max();
        TreeNode* bestNode = NULL;

        // iterate all immediate children and find best UTC score
        int childrenCount = node->getChildrenCount();
        for(int i = 0; i < childrenCount; i++) {
            TreeNode* child = node->getChild(i);
            float UCT_exploitation = (float)child->getValue() / (child->getVisitCount() + FLT_EPSILON);
            float UCT_exploration = sqrt( log((float)node->getVisitCount() + 1) / (child->getVisitCount() + FLT_EPSILON) );
            float UCT_score = UCT_exploitation + UCT_K * UCT_exploration;

            if(UCT_score > bestUctScore) {
                bestUctScore = UCT_score;
                bestNode = child;
            }
        }

        return bestNode;
    }

    TreeNode* getMostVisitedChild(TreeNode* node) const {
        int mostVisits = -1;
        TreeNode* bestNode = NULL;

        int childrenCount = node->getChildrenCount();
        for(int i = 0; i < childrenCount; i++) {
            TreeNode* child = node->getChild(i);
            if(child->getVisitCount() > mostVisits) {
                mostVisits = child->getVisitCount();
                bestNode = child;
            }
        }

        return bestNode;
    }

    Action run(const State& startingState) {
        // initialize timer
        timer.init();

        // initialize root TreeNode with current state
        TreeNode rootNode(startingState);

        TreeNode* bestNode = NULL;

        iterations = 0;
        while(true) {
            // indicate start of loop
            timer.loop_start();

            // 1. SELECT. Start at root, dig down into tree using UCT on all fully expanded nodes
            TreeNode* node = &rootNode;
            while(!node->isTerminalNode() && node->isFullyExpanded()) {
                node = getBestUctChild(node, UCT_K);
//                      assert(node);   // sanity check
            }

            // 2. EXPAND by adding a single child (if not terminal or not fully expanded)
            if(!node->isFullyExpanded() && !node->isTerminalNode()) node = node->expand();

            State state(node->getState());

            const std::vector<float> rewards = state.evaluate();

            node->visit();

            bestNode = getMostVisitedChild(&rootNode);

            // indicate end of loop for timer
            timer.loop_end();

            // exit loop if current total run duration (since init) exceeds allocatedTime
            if(allocatedTime > 0 && timer.check_duration(allocatedTime)) break;

            // exit loop if current iterations exceeds max_iterations
            if(max_iterations > 0 && iterations > max_iterations) break;
            iterations++;
        }

        // return best node's action
        if(bestNode) return bestNode->get_action();

        // we shouldn't be here
        return Action();
    }


};

#endif
