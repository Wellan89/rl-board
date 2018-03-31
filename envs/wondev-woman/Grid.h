#ifndef GRID_H
#define GRID_H

#include <set>
#include <queue>
#include <algorithm>

#include "Point.h"
#include "constants.h"
#include "Direction.h"

using namespace std;

class Grid {

public:

    vector<vector<int>> map;

    Grid() {
        this->map.resize(MAXIMUM_MAP_SIZE);
        for (int i = 0; i < this->map.size(); ++i) {
            this->map[i].resize(MAXIMUM_MAP_SIZE);
            for (int j = 0; j < this->map[i].size(); ++j) {
                this->map[i][j] = 4;
            }
        }

        int randomMapIndex = rand() % 3;
        if (randomMapIndex == 0) {
            this->generateRandomMap();
        } else {
            vector<Point> mapConfiguration;
            if (randomMapIndex == 1) {
                mapConfiguration = {
                    Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0), Point(4, 0),
                    Point(0, 1), Point(1, 1), Point(2, 1), Point(3, 1), Point(4, 1),
                    Point(0, 2), Point(1, 2), Point(2, 2), Point(3, 2), Point(4, 2),
                    Point(0, 3), Point(1, 3), Point(2, 3), Point(3, 3), Point(4, 3),
                    Point(0, 4), Point(1, 4), Point(2, 4), Point(3, 4), Point(4, 4)
                };
            } else if (randomMapIndex == 2) {
                mapConfiguration = {
                                                           Point(3, 0),
                                              Point(2, 1), Point(3, 1), Point(4, 1),
                                 Point(1, 2), Point(2, 2), Point(3, 2), Point(4, 2), Point(5, 2),
                    Point(0, 3), Point(1, 3), Point(2, 3), Point(3, 3), Point(4, 3), Point(5, 3), Point(6, 3),
                                 Point(1, 4), Point(2, 4), Point(3, 4), Point(4, 4), Point(5, 4),
                                              Point(2, 5), Point(3, 5), Point(4, 5),
                                                           Point(3, 6)
                };
            }
            for (Point point: mapConfiguration) {
                this->map[point.x][point.y] = 0;
            }
        }
    }

    void generateRandomMap() {
        int iterations = 0;
        int cells = 25 + rand() % 10;
        int islands = 0;

        while ((iterations * 2 < cells || islands > 1) && iterations < 1000) {
            int x = rand() % GENERATED_MAP_SIZE;
            int y = rand() % GENERATED_MAP_SIZE;

            this->map[x][y] = 0;
            this->map[GENERATED_MAP_SIZE - 1 - x][y] = 0;

            islands = this->countIslands();
            iterations++;
        }
    }

    int countIslands() const {
        set<int> computed;

        int total = 0;

        for (int x = 0; x < this->map.size(); x++) {
            for (int y = 0; y < this->map[x].size(); y++) {
                Point p(x, y);
                if (!this->isInMap(p)) {
                    continue;
                }
                if (computed.find(p.toHash()) == computed.end()) {
                    total++;
                    queue<Point> fifo;
                    fifo.push(p);
                    while (!fifo.empty()) {
                        Point e = fifo.front();
                        fifo.pop();
                        for (Direction d : DIRECTION_VALUES) {
                            Point n = e.getNeighbor(d);
                            if (computed.find(n.toHash()) == computed.end() && this->isInMap(n)) {
                                fifo.push(n);
                            }
                        }
                        computed.insert(e.toHash());
                    }
                }
            }
        }
        return total;
    }

    bool isInMap(int x, int y) const {
        return this->get(x, y) < FINAL_HEIGHT;
    }

    bool isInMap(const Point& p) const {
        return this->isInMap(p.x, p.y);
    }

    int get(int x, int y) const {
        if (x < 0 || y < 0 || x >= this->map.size() || y >= this->map.size()) {
            return 4;
        }
        return this->map[x][y];
    }

    int get(const Point& p) const {
        return this->get(p.x, p.y);
    }

    void place(Point placeAt) {
        this->map[placeAt.x][placeAt.y]++;
    }
};

#endif
