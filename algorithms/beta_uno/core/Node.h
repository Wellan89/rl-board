#ifndef NODE_H
#define NODE_H

#include <memory>
#include <math.h>
#include <vector>
#include <algorithm>

template <class State, typename Action>
class Node {

public:
    Node(const State& state, Node* parent = NULL):
        state(state),
        action(),
        parent(parent),
        playerId(state.playerId()),
        visitCount(0),
        depth(parent ? parent->depth + 1 : 0)
    {}

    Node* expand() {

        if (isFullyExpanded()) {
            return NULL;
        }

        if (actions.empty()) {
            state.computePossileActions(actions);

            std::random_shuffle(actions.begin(), actions.end());
        }

        return addChild(actions[children.size()]);
    }

    void visit() {
        visitCount++;
        if (parent) {
            parent.visit();
        }
    }

    const std::vector<float> evaluate () {
        auto result = state.evaluate();
        value = result[playerId];
        return result;
    }

    const State& getState() const {
        return state;
    }

    const Action& getAction() const {
        return action;
    }

    bool isFullyExpanded() const {
        return !children.empty() && children.size() == actions.size();
    }

    bool isTerminalNode() const {
        return state.isGameOver();
    }

    int getVisitCount() const {
        return visitCount;
    }

    float getValue() const {
        return value;
    }

    int getDepth() const {
        return depth;
    }

    int getChildrenCount() const {
        return children.size();
    }

    Node* getChild(int i) const {
        return children[i].get();
    }
    Node* getParent() const {
        return parent;
    }

private:
    GameState state;
    Action action;
    Node* parent;
    int playerId;

    int visitCount;
    int depth;
    float value;

    std::vector<std::shared_ptr<Node<State, Action>>> children;
    std::vector<Action> actions;

    Node* addChild(const Action& newAction) {
        Node* childNode = new Node(state, this);

        childNode->action = newAction;
        childNode->state.applyAction(newAction);
        children.push_back(std::shared_ptr<Node<State, Action>>(childNode));

        return childNode;
    }

};

#endif
