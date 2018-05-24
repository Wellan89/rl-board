#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stc.h"

namespace py = pybind11;

class World
{
public:
    GameState state;
    bool player_lost[2] = { false, false };

    float play(int action1, int action2) {
        Command cmd1 = allCommands[action1];
        Command cmd2 = allCommands[action2];
        if (!state.myGrid.canApplyCommand(cmd1))
            player_lost[0] = true;
        if (!state.oppGrid.canApplyCommand(cmd2))
            player_lost[1] = true;
        if (player_lost[0] || player_lost[1]) {
            return 0.0f;
        }
        std::pair<float, float> scores = state.play(cmd1, cmd2);
        return scores.first;
    }
    bool player_won(int player) const {
        return player_lost[1 - player];
    }
    vector<int> compute_state(bool opponent_view) const {
        vector<int> features;
        features.reserve(NB_BLOCKS_KNOWN * 2 + NB_LINES * NB_COLUMNS * 2);
        for (const Block& block : state.nextBlocks) {
            features.push_back(block.colorA);
            features.push_back(block.colorB);
        }
        for (int grid_id = 0; grid_id < 2; grid_id++) {
            const Grid& grid = grid_id ? state.oppGrid : state.myGrid;
            for (int i = 0; i < NB_LINES; i++) {
                for (int j = 0; j < NB_COLUMNS; j++) {
                    features.push_back(grid[Pos(i, j)]);
                }
            }
        }
        return features;
    }
    int dummy_opp_solution(int nb_opp_simulations) const {
        int empty_col = 2;
        int max_rows_empty = 0;
        for (int j = 0; j < NB_COLUMNS; j++) {
            int rows_empty = 0;
            while (rows_empty < NB_LINES && state.oppGrid[Pos(rows_empty, j)] == EMPTY_CELL)
                ++rows_empty;
            if (rows_empty > max_rows_empty) {
                max_rows_empty = rows_empty;
                empty_col = j;
            }
        }
        switch (empty_col) {
        case 0:
            return 16;
        case 1:
            return 19;
        case 2:
            return 1;
        case 3:
            return 5;
        case 4:
            return 9;
        case 5:
            return 12;
        }
        return 1;
    }
};


PYBIND11_MODULE(stc_pybind, m) {
    m.def("srand", [](unsigned int seed) { srand(seed); });

    py::class_<World>(m, "World")
        .def(py::init<>())
        .def("play", &World::play)
        .def("player_won", &World::player_won)
        .def("compute_state", &World::compute_state)
        .def("dummy_opp_solution", &World::dummy_opp_solution);
}
