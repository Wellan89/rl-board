#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "csb.h"

namespace py = pybind11;

const vector<vector<Point>> BASE_CONFIGURATIONS = {
    {Point(10540, 5980), Point(3580, 5180), Point(13580, 7600), Point(12460, 1350)},
    {Point(13840, 5080), Point(10680, 2280), Point(8700, 7460), Point(7200, 2160), Point(3600, 5280)},
    {Point(7350, 4940), Point(3320, 7230), Point(14580, 7700), Point(10560, 5060), Point(13100, 2320), Point(4560, 2180)},
    {Point(11480, 6080), Point(9100, 1840), Point(5010, 5260)},
    {Point(3450, 7220), Point(9420, 7240), Point(5970, 4240), Point(14660, 1410)},
    {Point(8000, 7900), Point(13300, 5540), Point(9560, 1400), Point(3640, 4420)},
    {Point(13500, 2340), Point(12940, 7220), Point(5640, 2580), Point(4100, 7420)},
    {Point(6320, 4290), Point(7800, 860), Point(7660, 5970), Point(3140, 7540), Point(9520, 4380), Point(14520, 7780)},
    {Point(13920, 1940), Point(8020, 3260), Point(2670, 7020), Point(10040, 5970)},
    {Point(6000, 5360), Point(11300, 2820), Point(7500, 6940)},
    {Point(13040, 1900), Point(6560, 7840), Point(7480, 1360), Point(12700, 7100), Point(4060, 4660)},
    {Point(6280, 7760), Point(14100, 7760), Point(13880, 1220), Point(10240, 4920), Point(6100, 2200), Point(3020, 5190)},
    {Point(11203, 5425), Point(7259, 6656), Point(5425, 2838), Point(10323, 3366)},
};
const int CHECKPOINT_MAX_DEVIATION = 30;


class WorldRunner
{
public:
    World w;

    void generate_map() {
        w.circuit.nblaps = 3;
        w.circuit.cps.clear();

        int conf_id = rand() % BASE_CONFIGURATIONS.size();
        int checkpointCount = BASE_CONFIGURATIONS[conf_id].size();
        int offset = rand() % checkpointCount;
        for (int i = 0; i < checkpointCount; i++) {
            int idx = (i + offset) % checkpointCount;
            Checkpoint cp(
                i,
                BASE_CONFIGURATIONS[conf_id][idx].x + rand()%(CHECKPOINT_MAX_DEVIATION+1) - CHECKPOINT_MAX_DEVIATION/2,
                BASE_CONFIGURATIONS[conf_id][idx].y + rand()%(CHECKPOINT_MAX_DEVIATION+1) - CHECKPOINT_MAX_DEVIATION/2
            );
            cp.r = 200;
            w.circuit.cps.push_back(cp);
        }
    }

    WorldRunner() {
        generate_map();

        float distance_to_center = 500 + 1000 * rand() % 2;
        float cp0x = w.circuit.cp(0).x;
        float cp0y = w.circuit.cp(0).y;
        float angle = PI / 2 + atan2(w.circuit.cp(1).y - cp0y, w.circuit.cp(1).x - cp0x);
        float cos_angle = cos(angle);
        float sin_angle = sin(angle);
        w.pods[0].x = cp0x + cos_angle * distance_to_center;
        w.pods[0].y = cp0y + sin_angle * distance_to_center;
        w.pods[1].x = cp0x - cos_angle * distance_to_center;
        w.pods[1].y = cp0y - sin_angle * distance_to_center;
        w.pods[2].x = cp0x + cos_angle * (2000 - distance_to_center);
        w.pods[2].y = cp0y + sin_angle * (2000 - distance_to_center);
        w.pods[3].x = cp0x - cos_angle * (2000 - distance_to_center);
        w.pods[3].y = cp0y - sin_angle * (2000 - distance_to_center);
        for (int i = 0; i < 4; i++) {
            w.pods[i].id = i;
            w.pods[i].angle = (angle - PI / 2) * 180 / PI;
        }
    }
    void play(const vector<float>& action1, const vector<float>& action2) {
        Solution s1, s2;
        s1.mv[0].g1 = action1[0];
        s1.mv[0].g2 = action1[1];
        s1.mv[0].g3 = action1[2];
        s1.mv[1].g1 = action1[3];
        s1.mv[1].g2 = action1[4];
        s1.mv[1].g3 = action1[5];
        s2.mv[0].g1 = action2[0];
        s2.mv[0].g2 = action2[1];
        s2.mv[0].g3 = action2[2];
        s2.mv[1].g1 = action2[3];
        s2.mv[1].g2 = action2[4];
        s2.mv[1].g3 = action2[5];
        w.play(s1, s2);
    }
    bool player_won(int player) {
        if (w.pods[player*2].lap == w.circuit.nblaps || w.pods[player*2+1].lap == w.circuit.nblaps)
            return true;
        else if (w.pods[(1-player)*2].timeout < 0 && w.pods[(1-player)*2+1].timeout < 0)
            return true;
        return false;
    }
    vector<float> compute_state(bool opponent_view) {
        vector<float> features;
        features.reserve(66);
        features.push_back(w.circuit.nblaps);
        features.push_back(w.circuit.nbcp());
        for (int _i = 0; _i < 4; _i++) {
            int i = _i;
            if (opponent_view) {
                if (i < 2)
                    i += 2;
                else
                    i -= 2;
            }
            features.push_back(w.pods[i].x / 5000.0);
            features.push_back(w.pods[i].y / 5000.0);
            features.push_back(w.pods[i].vx / 5000.0);
            features.push_back(w.pods[i].vy / 5000.0);
            features.push_back(w.pods[i].angle / 360.0);
            features.push_back(w.pods[i].boost_available);
            features.push_back(w.pods[i].timeout / 100.0);
            features.push_back(w.pods[i].shield / 4.0);
            features.push_back(w.pods[i].lap);
            features.push_back(w.pods[i].ncpid);
            for (int j = 0; j < 3; j++) {
                int next_checkpoint_id_j = (w.pods[i].ncpid + j) % w.circuit.nbcp();
                features.push_back(w.circuit.cp(next_checkpoint_id_j).x / 5000.0);
                features.push_back(w.circuit.cp(next_checkpoint_id_j).y / 5000.0);
            }
        }
        return features;
    }
    vector<float> dummy_opp_solution(int nb_opp_simulations) {
        vector<float> s;
        s.reserve(6);
        if (nb_opp_simulations > 0) {
            // NB: On CodinGame, the average number of simulations for a turn is 12000 (in 75ms)
            Solution solution = runGenetic(w.reversed(), 8, 10, nb_opp_simulations);
            for (int i = 0; i < 2; i++) {
                s.push_back(solution.mv[i].g1);
                s.push_back(solution.mv[i].g2);
                s.push_back(solution.mv[i].g3);
            }
        } else {
            for (int i = 2; i < 4; i++) {
                s.push_back((w.pods[i].diffAngle(w.circuit.cp(w.pods[i].ncpid)) + 18.0) / 36.0);
                s.push_back(80.0 / MAX_THRUST);
                s.push_back(0.5);
            }
        }
        return s;
    }
};


PYBIND11_MODULE(csb_pybind, m) {
    m.def("srand", [](unsigned int seed) { srand(seed); });

    py::class_<WorldRunner>(m, "World")
        .def(py::init<>())
        .def_property_readonly("circuit", [](WorldRunner& wr) { return wr.w.circuit; })
        .def_property_readonly("pods", [](WorldRunner& wr) { return vector<Pod>(wr.w.pods, wr.w.pods + NBPOD); })
        .def("play", &WorldRunner::play)
        .def("player_won", &WorldRunner::player_won)
        .def("compute_state", &WorldRunner::compute_state)
        .def("dummy_opp_solution", &WorldRunner::dummy_opp_solution);

    py::class_<Circuit>(m, "Circuit")
        .def("cp", &Circuit::cp)
        .def("nbcp", &Circuit::nbcp);

    py::class_<Checkpoint>(m, "Checkpoint")
        .def_readwrite("id", &Checkpoint::id)
        .def_readwrite("x", &Checkpoint::x)
        .def_readwrite("y", &Checkpoint::y)
        .def_readwrite("r", &Checkpoint::r);

    py::class_<Pod>(m, "Pod")
        .def_readwrite("id", &Pod::id)
        .def_readwrite("x", &Pod::x)
        .def_readwrite("y", &Pod::y)
        .def_readwrite("r", &Pod::r)
        .def_readwrite("angle", &Pod::angle)
        .def_readwrite("shield", &Pod::shield)
        .def_readwrite("boost_available", &Pod::boost_available)
        .def("nb_checked", &Pod::nb_checked)
        .def("score", &Pod::env_score)
        .def("next_checkpoint", [](Pod& pod) { return pod.circuit->cp(pod.ncpid); });
}
