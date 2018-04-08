#pragma GCC optimize("-O3")
#pragma GCC optimize("inline")
#pragma GCC optimize("omit-frame-pointer")
#pragma GCC optimize("unroll-loops")

#define USE_TIME_LIMIT 0
#define USE_SIMULATION_LIMIT 1

// CONF

// TODO : en args du programme

const int limit_type = USE_TIME_LIMIT;

const float limit_seconds = 0.140;
const int limit_simulations = 10000;

// end CONF

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cmath>
#include <initializer_list>

#define PI 3.14159265358979323846
#define MAX_THRUST 200
#define NBPOD 4

using namespace std;
using namespace chrono;

int DEBUG_LEVEL = 0;

int nblaps;
int checkpointCount;

high_resolution_clock::time_point global_time;

static unsigned long x = 123456789, y = 362436069, z = 521288629;
unsigned long myrand(void) {
    unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;
    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;
    return z;
}

// TODO : Opti
#define LIN(x, x1, y1, x2, y2) (y1 + (y2-y1)*(x-x1)/(x2-x1))

inline void log(int lvl, string str) {
    if (DEBUG_LEVEL >= lvl)
        cerr << str;
}


class Move {
public:
    Move(){}
    float g1=0,g2=0,g3=0;
    void randomize() {
        g1 = (double) rand() / RAND_MAX;
        g2 = (double) rand() / RAND_MAX;
        g3 = (double) rand() / RAND_MAX;
    }
    void mutate_gene(float* g, float t) {
        float ramax = *g + 1.0 * t;
        float ramin = *g - 1.0 * t;
        if (ramax > 1)
            ramax = 1.0;
        if (ramin < 0)
            ramin = 0.0;
        *g = ramin + ((ramax-ramin)* (((float) rand()) / (float) RAND_MAX));
    }
    void mutate(float t) {
        //assert(t > 0 && t <= 1),
        mutate_gene(&g1,t);
        mutate_gene(&g2,t);
        mutate_gene(&g3,t);
    }
};

class Solution {
public:
    Solution(){}
    Move mv[2];
    void randomize() {
        mv[0].randomize();
        mv[1].randomize();
    }
    void mutate(float t) {
        //assert(t >= 0 && t <= 1.0);
        int idm = rand()%2;
        mv[idm].mutate(t);
    }
};

class Point {
public:
    float x;
    float y;
    Point(){}
    Point(float _x, float _y):x(_x),y(_y){}
    float distance2(Point p) {
        return (this->x - p.x)*(this->x - p.x)+(this->y - p.y)*(this->y - p.y);
    }
    float distance(Point p) {
        return sqrt(this->distance2(p));
    }
    Point closest(Point a, Point b) {
        float da = b.y - a.y;
        float db = a.x - b.x;
        float c1 = da*a.x + db*a.y;
        float c2 = -db*this->x + da*this->y;
        float det = da*da + db*db;
        float cx = 0;
        float cy = 0;
        if (det != 0) {
            cx = (da*c1 - db*c2) / det;
            cy = (da*c2 + db*c1) / det;
        } else {
            cx = this->x;
            cy = this->y;
        }
        return Point(cx, cy);
   }
};

class Unit;

class Collision
{
public:
    Collision(Unit* _a, Unit* _b, float _t):a(_a),b(_b),t(_t){}
    Unit* a;
    Unit* b;
    float t;
};

class Unit : public Point
{
public:
    Unit(){}
    int id=-1;
    float r=-1;
    float vx=0;
    float vy=0;

    Unit(int _id, float _x, float _y, float _r, float _vx, float _vy):
        Point(_x,_y),
        id(_id),
        r(_r),
        vx(_vx),vy(_vy)
    {}

    Collision* collision(Unit* e) {
        //assert(e != NULL);
        float dist = this->distance2(*e);
        //assert(e->r > 0);
        //assert(this->r > 0);
        float somme_rayon_2 = (this->r+e->r)*(this->r+e->r);
        if (dist < somme_rayon_2) {
            return new Collision(this, e, 0.0);
        }

        if (this->vx == e->vx && this->vy == e->vy)
            return NULL;

        float x = this->x - e->x;
        float y = this->y - e->y;
        Point myp = Point(x,y);
        float vx = this->vx - e->vx;
        float vy = this->vy - e->vy;
        Point up = Point(0.0,0.0);
        Point p = up.closest(myp, Point(x+vx, y+vy));

        float pdist = up.distance2(p);
        float mypdist = myp.distance2(p);

        if (pdist >= somme_rayon_2)
            return NULL;

        float length = sqrt(vx*vx+vy*vy);
        float backdist = sqrt(somme_rayon_2 - pdist);
        p.x = p.x - backdist * (vx / length);
        p.y = p.y - backdist * (vy / length);

        if (myp.distance2(p) > mypdist)
            return NULL;

        pdist = p.distance(myp);

        if (pdist >= length)
            return NULL;

        return new Collision(this, e, pdist/length);
    }

    virtual void bounce(Unit* e)=0;
};

class Checkpoint : public Unit
{
public:
    Checkpoint(int _id, int _x, int _y):
        Unit(_id,_x,_y,200.0,0.0,0.0){}
    void bounce(Unit* u);
};

class Circuit
{
public:
    int nbcp() {
        return cps.size();
    }
    Checkpoint& cp(int cpid) {
        return cps[cpid];
    }
    vector<Checkpoint> cps;
};
Circuit circuit;

class Pod : public Unit
{
public:
    Pod(){r=400.0;}
    Pod(int _id, int _x, int _y):
        Unit(_id,_x,_y,400.0,0.0,0.0){}
    float angle=0.0;
    int ncpid=1;
    int timeout=100;
    int shield=0;
    int lap=0;
    bool boost_available=true;

    /*bool save_boost_available;
    int save_lap;
    int save_shield;
    float save_angle;
    int save_ncpid;
    int save_timeout;
    float save_x, save_y, save_vx, save_vy;

    void save() {
        save_boost_available = boost_available;
        save_lap = lap;
        save_shield = shield;
        save_angle = angle;
        save_ncpid = ncpid;
        save_timeout = timeout;
        save_x = x;
        save_y = y;
        save_vx = vx;
        save_vy = vy;
    }

    void rollback() {
        boost_available = save_boost_available;
        lap = save_lap;
        shield = save_shield;
        angle = save_angle;
        ncpid = save_ncpid;
        timeout = save_timeout;
        x = save_x;
        y = save_y;
        vx = save_vx;
        vy = save_vy;
    }*/

    bool compare(Pod& p) {
        if (abs(x-p.x)>1 || abs(y-p.y)>1 || abs(vx-p.vx)>1 || abs(vy-p.vy)>1 || ncpid != p.ncpid)
            return false;
        return true;
    }
    void show() {
        cerr << "Pod " << id << " : (" << x << "," << y << ") v(" << vx << "," << vy << ") ang: " << angle << " ncpid:" << ncpid << endl;
    }
    void incrcpid() {
        ncpid++;
        if (ncpid == 1)
            lap++;
        if (ncpid >= checkpointCount) {
            ncpid = 0;
        }
    }
    float score() {
        int lastCP = ncpid-1;
        if (lastCP == -1)
            lastCP = checkpointCount-1;
        int nbChecked = lap*checkpointCount+lastCP;
        return nbChecked*50000-distance(circuit.cp(ncpid));
    }
    float getAngle(Point p) {
        float d = this->distance(p);
        float dx = (p.x - this->x) / d;
        float dy = (p.y - this->y) / d;
        float a = acos(dx)*180.0/PI;
        if (dy < 0)
            a = 360.0-a;
        return a;
    }
    float diffAngle(Point p) {
        float a = this->getAngle(p);
        float right = this->angle <= a ? a-this->angle : 360.0-this->angle+a;
        float left = this->angle >= a ? this->angle-a : this->angle+360.0-a;
        if (right < left)
            return right;
        return -left;
    }

    void rotate(Point p) {
        float a = this->diffAngle(p);
        if (a > 18.0) {
            a = 18.0;
        } else if (a < -18.0) {
            a = -18.0;
        }
        this->angle += a;
        if (this->angle >= 360.0)
            this->angle -= 360.0;
        else if (this->angle < 0.0)
            this->angle += 360.0;
    }
    void boost(int thrust) {
        //log(4,"Pod::boost IN\n");
        if (this->shield > 0) {
            return;
        }
        float ra = this->angle * PI / 180.0;
        //log(5,"Pod::boost angle : " + to_string(ra) + ", thrust" + to_string(thrust) + " on v("+ to_string(this->vx) + "," + to_string(this->vy) + ")\n");
        this->vx += cos(ra) * thrust;
        this->vy += sin(ra) * thrust;
        //log(5,"Pod::boost new velocity : v("+ to_string(this->vx) + "," + to_string(this->vy) + ")\n");
        //log(4,"Pod::boost OUT\n");
    }
    void move(float t) {
        //log(4,"Moving pod from(" + to_string(this->x) + "," + to_string(this->y) + ") ");
        this->x += this->vx*t;
        this->y += this->vy*t;
        //log(4,"to (" + to_string(this->x) + "," + to_string(this->y) + ") (time " + to_string(t) + ")\n");
    }
    void end() {
        //log(4,"Pod::end IN\n");

        //log(5,"Pod::end before : (" + to_string(this->x) + "," + to_string(this->y) + ") (" +to_string(this->vx) + "," + to_string(this->vy) + ")\n");
        this->x = round(this->x);
        this->y = round(this->y);
        this->vx = static_cast<int>(this->vx * 0.85);
        this->vy = static_cast<int>(this->vy * 0.85);
        this->timeout -= 1;
        //log(5,"Pod::end after : (" + to_string(this->x) + "," + to_string(this->y) + ") (" +to_string(this->vx) + "," + to_string(this->vy) + "\n");
        //log(4,"Pod::end OUT\n");
    }
    /*void play(Point p, int thrust) {
        this->rotate(p);
        this->boost(thrust);
        this->move(1.0);
        this->end();
    }*/
    void bounce(Unit* e) {
        //assert(e != NULL);
        float tm = this->shield==4?10:1;
        float em = static_cast<Pod*>(e)->shield==4?10:1;
        float mcoeff = (tm + em) / (tm * em);
        float nx = this->x - e->x;
        float ny = this->y - e->y;
        float nxnysquare = nx*nx + ny*ny;
        float dvx = this->vx - e->vx;
        float dvy = this->vy - e->vy;
        float product = nx*dvx + ny*dvy;
        //assert(nxnysquare > 0);
        float fx = (nx * product) / (nxnysquare* mcoeff);
        float fy = (ny * product) / (nxnysquare* mcoeff);

        this->vx -= fx / tm;
        this->vy -= fy / tm;
        e->vx += fx / em;
        e->vy += fy / em;

        float impulse = sqrt(fx*fx + fy*fy);
        //assert(impulse != 0);
        if (impulse < 120.0) {
            fx = fx * 120.0 / impulse;
            fy = fy * 120.0 / impulse;
        }

        this->vx -= fx / tm;
        this->vy -= fy / tm;
        e->vx += fx / em;
        e->vy += fy / em;
    }

    float get_new_angle(float gene) {
        //assert(gene <= 1 && gene >= 0);
        float res = this->angle;
        if (gene < 0.25) {
            res -= 18.0;
        } else if (gene > 0.75) {
            res += 18.0;
        } else {
            res += LIN(gene,0.25,-18.0,0.75,18.0);
        }
        if (res >= 360.0) {
            res -= 360.0;
        } else if (res < 0.0) {
            res += 360.0;
        }
        return res;
    }

    float get_new_power(float gene) {
        //assert(gene <= 1 && gene >= 0);
        if (gene < 0.2) {
            return 0;
        } else if (gene > 0.8) {
            return MAX_THRUST;
        } else {
            return LIN(gene,0.2,0,0.8,MAX_THRUST);
        }
    }

    // précond : pas un doombot move
    void apply_move(Move& mv) {
        this->angle = get_new_angle(mv.g1);
        if (mv.g3 < 0.05 && boost_available) {
            if (this->shield == 0) {
                boost_available = false;
                boost(650);
            }
        } else if (mv.g3 > 0.95) {// no shield
            this->shield = 4;
        } else {
            if (this->shield == 0)
                boost(get_new_power(mv.g2));
        }
    }
    void output(Move& mv) {
        float a = get_new_angle(mv.g1) * PI / 180.0;
        //assert(a >= 0.0 && a <= 2*PI);
        int px = this->x + cos(a) * 1000000.0;
        int py = this->y + sin(a) * 1000000.0;
        float power = get_new_power(mv.g2);
        //assert(power >= 0 && power <= MAX_THRUST);
        string msg = ":";
        if (mv.g3 < 0.05 && boost_available)
            cout << px << " " << py << " BOOST " << msg << endl;
        else if (mv.g3 > 0.95)
            cout << px << " " << py << " SHIELD " << msg << endl;
        else
            cout << px << " " << py << " " << round(power) << " " << msg << endl;
    }
};

void Checkpoint::bounce(Unit* u) {
    Pod* pod = static_cast<Pod*>(u);
    pod->incrcpid();
    pod->timeout = 100;
}

Move convert_to_move(Pod p, Point& dest) {
    Move mv;
    mv.g3 = 0.5;
    mv.g2 = 1.0;
    float diffA = p.diffAngle(dest);
    mv.g1 = LIN(diffA,-18.0,0.25,18.0,0.75);
    if (mv.g1 > 1)
        mv.g1 = 1;
    if (mv.g1 < 0)
        mv.g1 = 0;
    return mv;
}



class World
{
public:
    World(){}
    Pod pods[NBPOD];

    Pod save_pods[NBPOD];

    void save() {
        for (int i = 0; i < NBPOD; i++) {
            // pods->save();
            save_pods[i] = pods[i];
        }
    }
    void rollback() {
        for (int i = 0; i < NBPOD; i++) {
           // pods->rollback();
            pods[i] = save_pods[i];
        }
    }

    void play(Solution& s1, Solution& s2) {
        for (int i = 0; i < NBPOD; i++) {
            //assert(pods[i].shield >= 0 && pods[i].shield <= 4);
            //assert(pods[i].ncpid >= 0 && pods[i].ncpid <= checkpointCount-1);
            //assert(pods[i].angle >= 0.0 && pods[i].angle <= 360.0);
            if (pods[i].shield > 0)
                pods[i].shield--;
        }
        pods[0].apply_move(s1.mv[0]);
        pods[1].apply_move(s1.mv[1]);
        pods[2].apply_move(s2.mv[0]);
        pods[3].apply_move(s2.mv[1]);

        float t = 0.0;
        bool previousCollision = false;
        Unit* lasta = NULL;
        Unit* lastb = NULL;
        while (t < 1.0) {
            Collision firstCol(NULL, NULL, -1.0);
            bool foundCol = false;
            for (int i = 0; i < NBPOD; i++) {
                for(int j = i+1; j < NBPOD; j++) {
                    Collision* col = pods[i].collision(&pods[j]);
                    if (col != NULL) {
                        if (col->t + t < 1.0 && (!foundCol || col->t < firstCol.t)) {
                            firstCol = *col;
                            foundCol = true;
                        }
                        delete col;
                    }
                }
                Collision* col = pods[i].collision(&(circuit.cps[pods[i].ncpid]));
                if (col != NULL) {
                    if (col->t + t < 1.0 && (!foundCol || col->t < firstCol.t)) {
                        firstCol = *col;
                        foundCol = true;
                    }
                    delete col;
                }
            }

            if (!foundCol || (previousCollision && firstCol.t == 0.0 && firstCol.a == lasta && firstCol.b == lastb)) {
                for (int i = 0; i < NBPOD; i++) {
                    pods[i].move(1.0-t);
                }
                t = 1.0;
            } else {
                //assert(firstCol.b != NULL);
                //assert(firstCol.a != NULL);
                //assert(firstCol.t != -1.0);
                previousCollision = true;
                lasta = firstCol.a;
                lastb = firstCol.b;
                for (int i = 0; i < NBPOD; i++) {
                    pods[i].move(firstCol.t);
                }
                firstCol.b->bounce(firstCol.a);
                t += firstCol.t;
            }

        } // while (t < 1.0)

        for (int i = 0; i < NBPOD; i++) {
            pods[i].end();
        }
    }

    bool compare(World& w) {
        bool foundDifference = false;
        for (int i = 0; i < NBPOD; i++) {
            if (!pods[i].compare(w.pods[i])) {
                cerr << "Expected pod : ";
                w.pods[i].show();
                cerr << "Received input pod : ";
                pods[i].show();
                foundDifference = true;
            }
        }
        return foundDifference;
    }

    float eval(int player_id) {
        int player_pod_0 = player_id == 0 ? 0 : 2;
        int player_pod_1 = player_id == 0 ? 1 : 3;
        int enemy_pod_0 = player_id == 0 ? 2 : 0;
        int enemy_pod_1 = player_id == 0 ? 3 : 1;

        if (pods[enemy_pod_0].lap == nblaps || pods[enemy_pod_1].lap == nblaps)
            return -(numeric_limits<float>::max()/2);
        else if (pods[player_pod_0].lap == nblaps || pods[player_pod_1].lap == nblaps)
            return (numeric_limits<float>::max()/2);
        float s1 = pods[player_pod_0].score();
        float s2 = pods[player_pod_1].score();
        float e1 = pods[enemy_pod_0].score();
        float e2 = pods[enemy_pod_1].score();
        float mrs,ers;
        int my_chaser, his_runner;
        if (s1 > s2) {
            mrs = s1;
            my_chaser = player_pod_1;
        } else {
            mrs = s2;
            my_chaser = player_pod_0;
        }
        if (e1 > e2) {
            ers = e1;
            his_runner = enemy_pod_0;
        } else {
            ers = e2;
            his_runner = enemy_pod_1;
        }
        float ret = 50 * (mrs - ers);
        ret -= pods[my_chaser].distance(circuit.cp(pods[his_runner].ncpid));
        ret -= pods[my_chaser].diffAngle(pods[his_runner]);
        return ret;
    }

    int winner() {
        if (pods[0].timeout <= 0 && pods[1].timeout <= 0) {
            return 1;
        }
        if (pods[2].timeout <= 0 && pods[3].timeout <= 0) {
            return 0;
        }
        for (int i = 0; i < NBPOD; i++) {
            if (pods[i].lap >= nblaps) {
                return (i >= 2);
            }
        }
        return -1;
    }
};

Solution* enemyMoves;

inline bool should_continue(float current_time, float allocated_time, int simulation_count, int simulation_limit) {
    if (limit_type == USE_TIME_LIMIT) {
        return current_time < allocated_time;
    } else if (limit_type == USE_SIMULATION_LIMIT) {
        return simulation_count < simulation_limit;
    } else {
        cout << "Attention mon coco" << endl;
        return false;
    }
}

Solution genetic(int profondeur, int population, float allocated_time, int simulation_limit,
                 World& world, Solution* otherPlayer, bool output, int player_id) {
    //assert(profondeur > 0);
    //assert(allocated_time > 0.0);
    //assert(population > 0);
    world.save();
    int cnt = 0;

    high_resolution_clock::time_point starting_time = high_resolution_clock::now();

    Solution pop[population][profondeur];
    float evals[population];

    // Generate initial population
    for (int i = 0; i < population; i++) {
        for (int j = 0; j < profondeur; j++) {
            pop[i][j].randomize();
        }
    }

    // Eval initial population
    for (int i = 0; i < population; i++) {
        world.rollback();
        for (int j = 0; j < profondeur; j++) {
            if (player_id == 0) {
                world.play(pop[i][j], otherPlayer[j]);
            } else {
                world.play(otherPlayer[j], pop[i][j]);
            }
        }
        evals[i] = world.eval(player_id);
    }

    // Find solution to replace
    float current_min = numeric_limits<float>::max();
    int current_min_id=-1;
    for (int i = 0; i < population; i++) {
        if (evals[i] < current_min) {
            current_min = evals[i];
            current_min_id = i;
        }
    }
    //assert(current_min_id >= 0 && current_min_id < population);

    float mutate_range;

    high_resolution_clock::time_point currtime = high_resolution_clock::now();
    while (should_continue(
        duration_cast<duration<double>> (currtime - starting_time).count(),
        allocated_time,
        cnt,
        simulation_limit
    )) {
        cnt++;
        world.rollback();
        Solution newsol[profondeur];

        int randid = rand()%population;
        for (int i = 0; i < profondeur; i++) {
            newsol[i] = pop[randid][i];
        }

        if (limit_type == USE_TIME_LIMIT) {
            mutate_range = 1.0 - (
                duration_cast<duration<double>> (
                    currtime - starting_time
                ).count()
            ) / allocated_time;
        } else if (limit_type == USE_SIMULATION_LIMIT) {
            mutate_range = 1.0 - (double)cnt / (double)simulation_limit;
        }

        for (int i = 0; i < profondeur; i++) {
            newsol[i].mutate(mutate_range);
            if (player_id == 0) {
                world.play(newsol[i], otherPlayer[i]);
            } else {
                world.play(otherPlayer[i], newsol[i]);
            }
        }

        float current_eval = world.eval(player_id);

        if (current_eval > current_min) {
            current_min = current_eval;
            for (int j = 0; j < profondeur; j++) {
                pop[current_min_id][j] = newsol[j];
            }
            evals[current_min_id] = current_eval;
            for (int i = 0; i < population; i++) {
                if (evals[i] < current_min) {
                    current_min = evals[i];
                    current_min_id = i;
                }
            }
            //assert(current_min_id >= 0 && current_min_id < population);
        }
        currtime = high_resolution_clock::now();
    }

    // Find best in pop
    int better_sol_id = -1;
    float current_max = -numeric_limits<float>::max();
    for (int i = 0; i < population; i++) {
        if (evals[i] > current_max) {
            current_max = evals[i];
            better_sol_id = i;
        }
    }
    //assert(better_sol_id >= 0 && better_sol_id < population);
    world.rollback();
    if (!output) {
        for (int i = 0; i < profondeur; i++) {
            enemyMoves[i] = pop[better_sol_id][i];
        }
    }
    return pop[better_sol_id][0];
}

class Runner {

    int run_nb = 0;
    int turn = 0;
    int prof = 8;
    int pop = 10;

    World w;

    ofstream output_file;

public:

    Runner() {}

    void reset() {
        run_nb++;

        if (output_file.is_open()) {
            output_file.close();
        }
        output_file.open("games/" + to_string(run_nb) + "_" + get_adjective() + "_" + get_noun() + "_" + to_string(rand() % 1000) + ".game");

        turn = 0;
        enemyMoves = new Solution[prof];
        generate_map();
        w = World();

        int distance_to_center = (rand()%2) ? 1500: 500;
        int cp0x = circuit.cp(0).x;
        int cp0y = circuit.cp(0).y;
        float angle = PI / 2 + atan2(circuit.cp(1).y - cp0y, circuit.cp(1).x - cp0x);
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

        for (int i = 0; i < NBPOD; i++) {
            w.pods[i].angle = angle * 180.0 / PI;
            if (w.pods[i].angle < 0) {
                w.pods[i].angle += 360;
            }
            w.pods[i].id = i;
            w.pods[i].lap = 0;
            w.pods[i].boost_available = true;
            w.pods[i].ncpid = 1;
            w.pods[i].timeout = 100;
            w.pods[i].shield = 0;
        }
    }

    string get_adjective() {
        vector<string> adjectives = {"fabulous", "incredible", "wonderful", "great", "delightful",
            "marvelous", "unbelieveable", "enormous", "sad", "happy", "pretty", "ugly", "good", "bad",
            "joyful", "little", "beautiful", "angry", "big", "scary", "formidable", "dynamic",
            "enlighted", "bored", "intelligent", "colorful"
        };
        return adjectives[rand() % adjectives.size()];
    }

    string get_noun() {
        vector<string> nouns = {"coconut", "abricot", "peach", "fruit", "apple", "orange", "banana",
            "sun", "moon", "beast", "horse", "pellican", "lion", "tiger", "elephant", "baby", "chicken",
            "mouse", "spoon", "sponge", "tuperware", "poney", "kassadin", "evelynn", "bird", "vincent", "salim",
            "destygo", "chocolat", "coffee", "python", "movie", "lizard", "fog", "snake", "fish", "cow",
            "pineaple", "mountain", "tulipe", "flower"
        };
        return nouns[rand() % nouns.size()];
    }

    void generate_map() {
        nblaps = 3;
        circuit.cps = vector<Checkpoint>();

        int CHECKPOINT_MAX_DEVIATION = 30;

        vector<vector<Point>> BASE_CONFIGURATIONS = {
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

        int conf_id = rand() % BASE_CONFIGURATIONS.size();

        checkpointCount = BASE_CONFIGURATIONS[conf_id].size();
        int offset = rand() % checkpointCount;
        for (int i = 0; i < checkpointCount; i++) {
            int idx = (i + offset) % checkpointCount;
            circuit.cps.push_back(
                Checkpoint(
                    i,
                    BASE_CONFIGURATIONS[conf_id][idx].x + rand()%(CHECKPOINT_MAX_DEVIATION+1) - CHECKPOINT_MAX_DEVIATION/2,
                    BASE_CONFIGURATIONS[conf_id][idx].y + rand()%(CHECKPOINT_MAX_DEVIATION+1) - CHECKPOINT_MAX_DEVIATION/2
                )
            );
        }
    }

    void run() {
        while (1) {
            reset();
            run_and_save_game();
            cout << "Generated game " << run_nb << endl;
        }
    }

    void run_and_save_game() {
        write_map_info();
        while (w.winner() == -1 && turn < 500) {
            turn++;
            write_pods();
            output_file << w.eval(0) << " " << w.eval(1) << endl;
            Solution s1 = get_solution(0);
            Solution s2 = get_solution(1);
            write_solution(s1);
            write_solution(s2);
            w.play(s1, s2);
        }
        output_file << w.winner() << endl;
    }

    void write_map_info() {
        output_file << nblaps << " " << checkpointCount << endl;
        for (int i = 0; i < checkpointCount; i++) {
            output_file << circuit.cp(i).x << " " << circuit.cp(i).y << endl;
        }
    }

    void write_solution(const Solution& s) {
        output_file << s.mv[0].g1 << " " << s.mv[0].g2 << " " << s.mv[0].g3 << " " << s.mv[1].g1 << " " << s.mv[1].g2 << " " << s.mv[1].g3 << endl;
    }

    void write_pod(const Pod& p) {
        output_file << p.x << " " << p.y << " " << p.vx << " " << p.vy << " " << p.angle << " " << p.ncpid << " " << p.lap << " " << p.timeout << " " << p.shield << " " << p.boost_available << endl;
    }

    void write_pods() {
        output_file << turn << endl;
        for (int i = 0; i < NBPOD; i++) {
            write_pod(w.pods[i]);
        }
    }

    Solution get_solution(int player_id) {
        int player_pod_0 = player_id == 0 ? 0 : 2;
        int player_pod_1 = player_id == 0 ? 1 : 3;
        int enemy_pod_0 = player_id == 0 ? 2 : 0;
        int enemy_pod_1 = player_id == 0 ? 3 : 1;

        Solution emptyMoves[prof];
        Solution otherPlayer[prof];
        World doom_bot = w;
        for (int i = 0; i < prof; i++) {
            Point d1, d2;
            if (doom_bot.pods[player_pod_0].score() > doom_bot.pods[player_pod_1].score()) {
                d1 = circuit.cp(doom_bot.pods[player_pod_0].ncpid);
                d2 = (doom_bot.pods[enemy_pod_0].score() > doom_bot.pods[enemy_pod_1].score())?doom_bot.pods[enemy_pod_0]:doom_bot.pods[enemy_pod_1];
            } else {
                d1 = (doom_bot.pods[enemy_pod_0].score() > doom_bot.pods[enemy_pod_1].score())?doom_bot.pods[enemy_pod_0]:doom_bot.pods[enemy_pod_1];
                d2 = circuit.cp(doom_bot.pods[player_pod_1].ncpid);
            }
            otherPlayer[i].mv[0] = convert_to_move(doom_bot.pods[player_pod_0], d1);
            otherPlayer[i].mv[1] = convert_to_move(doom_bot.pods[player_pod_1], d2);
            // doom_bot.play(otherPlayer[i], emptyMoves[i]);
        }
        World reversedWorld = w;
        Pod tmp = w.pods[player_pod_0];
        reversedWorld.pods[player_pod_0] = w.pods[enemy_pod_0];
        reversedWorld.pods[enemy_pod_0] = tmp;
        tmp = w.pods[player_pod_1];
        reversedWorld.pods[player_pod_1] = w.pods[enemy_pod_1];
        reversedWorld.pods[enemy_pod_1] = tmp;
        genetic(prof, pop, limit_seconds/15, limit_simulations/15, reversedWorld, otherPlayer, false, player_id);
        for (int i = 0; i < prof; i++) {
            otherPlayer[i] = enemyMoves[i];
        }
        return genetic(prof, pop, limit_seconds - (0.003 + limit_seconds/15), limit_simulations, w, otherPlayer, true, player_id);
    }
};

int main()
{
    srand(time(NULL));
    Runner runner;
    runner.run();
}
