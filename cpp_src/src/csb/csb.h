#ifndef DEF_CSB_H
#define DEF_CSB_H

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
//#include <assert.h>
#include <cmath>

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
        Unit(_id,_x,_y,-100.0,0.0,0.0){}
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
    int nb_checked() {
        int lastCP = ncpid - 1;
        if (lastCP == -1)
            lastCP = circuit.nbcp() - 1;
        return lap * circuit.nbcp() + lastCP;
    }
    float score() {
        return nb_checked() * 50000 - distance(circuit.cp(ncpid));
    }
    float env_score() {
        Checkpoint& current_cp = circuit.cp((ncpid + circuit.nbcp() - 1) % circuit.nbcp());
        Checkpoint& next_cp = circuit.cp(ncpid);
        float distance_cp_to_ncp = current_cp.distance(next_cp);
        float cp_dist_score = (distance_cp_to_ncp - distance(next_cp)) / distance_cp_to_ncp;
        return nb_checked() + cp_dist_score;
    }
    float getAngle(Point p) {
        return atan2(p.y - y, p.x - x) * 180.0 / PI;
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
        vector<Unit*> lasta;
        vector<Unit*> lastb;
        while (t < 1.0) {
            Collision firstCol(NULL, NULL, -1.0);
            bool foundCol = false;
            for (int i = 0; i < NBPOD; i++) {
                for(int j = i+1; j < NBPOD; j++) {
                    if ((std::find(lasta.begin(), lasta.end(), &pods[i]) != lasta.end()) && (std::find(lastb.begin(), lastb.end(), &pods[j]) != lastb.end())) {
                        continue;
                    }
                    Collision* col = pods[i].collision(&pods[j]);
                    if (col != NULL) {
                        if (col->t + t < 1.0 && (!foundCol || col->t < firstCol.t)) {
                            firstCol = *col;
                            foundCol = true;
                        }
                        delete col;
                    }
                }
                if ((std::find(lasta.begin(), lasta.end(), &pods[i]) != lasta.end()) && (std::find(lastb.begin(), lastb.end(), &(circuit.cps[pods[i].ncpid])) != lastb.end())) {
                    continue;
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

            if (!foundCol) {
                for (int i = 0; i < NBPOD; i++) {
                    pods[i].move(1.0-t);
                }
                t = 1.0;
            } else {
                if (firstCol.t > 0.0) {
                    lasta.clear();
                    lastb.clear();
                }
                //assert(firstCol.b != NULL);
                //assert(firstCol.a != NULL);
                //assert(firstCol.t != -1.0);
                previousCollision = true;
                lasta.push_back(firstCol.a);
                lastb.push_back(firstCol.b);
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

    float eval() {
        if (pods[2].lap == nblaps || pods[3].lap == nblaps)
            return -(numeric_limits<float>::max()/2);
        else if (pods[0].lap == nblaps || pods[1].lap == nblaps)
            return (numeric_limits<float>::max()/2);
        float s1 = pods[0].score();
        float s2 = pods[1].score();
        float e1 = pods[2].score();
        float e2 = pods[3].score();
        float mrs,ers;
        int my_chaser, his_runner;
        if (s1 > s2) {
            mrs = s1;
            my_chaser = 1;
        } else {
            mrs = s2;
            my_chaser = 0;
        }
        if (e1 > e2) {
            ers = e1;
            his_runner = 2;
        } else {
            ers = e2;
            his_runner = 3;
        }
        float ret = 50*(mrs - ers);
        ret -= pods[my_chaser].distance(circuit.cp(pods[his_runner].ncpid));
        ret -= pods[my_chaser].diffAngle(pods[his_runner]);
        return ret;
    }
};

World expected;

void monte_carlo(int profondeur, float allocated_time, World& world, Solution* otherPlayer) {
    //assert(profondeur > 0);
    //assert(allocated_time > 0.0);
    int cnt = 0;

    high_resolution_clock::time_point starting_time = high_resolution_clock::now();

    Solution best_decision;
    float best_eval = -numeric_limits<float>::max();
    while (duration_cast<duration<double>>
               (high_resolution_clock::now() - starting_time).count()
                        < allocated_time)
    {
        cnt++;
        World world_instance = world;
        Solution firstDec, nextDec;
        firstDec.randomize();
        world_instance.play(firstDec, otherPlayer[0]);

        for (int i = 1; i < profondeur; i++) {
            nextDec.randomize();
            world_instance.play(nextDec, otherPlayer[i]);
        }

        float current_eval = world_instance.eval();

        if (current_eval > best_eval) {
            best_eval = current_eval;
            best_decision = firstDec;
        }
    }
    std::cerr << cnt << std::endl;
    world.pods[0].output(best_decision.mv[0]);
    world.pods[1].output(best_decision.mv[1]);
    expected = world;
    expected.play(best_decision, otherPlayer[0]);
}

Solution* enemyMoves;

void genetic(int profondeur, int population, float allocated_time, World& world, Solution* otherPlayer, bool output) {
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
            world.play(pop[i][j],otherPlayer[j]);
        }
        evals[i] = world.eval();
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


    high_resolution_clock::time_point currtime = high_resolution_clock::now();
    while (duration_cast<duration<double>> (currtime - starting_time).count() < allocated_time)
    {
        cnt++;
        world.rollback();
        Solution newsol[profondeur];

        int randid = rand()%population;
        for (int i = 0; i < profondeur; i++) {
            newsol[i] = pop[randid][i];
        }

        for (int i = 0; i < profondeur; i++) {
            newsol[i].mutate(
                1.0 - (
                    duration_cast<
                        duration<double>
                    > (
                        currtime - starting_time
                    ).count()
                      ) / allocated_time
            );
            world.play(newsol[i], otherPlayer[i]);
        }

        float current_eval = world.eval();

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
    std::cerr << "cnt : " << cnt << std::endl;

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
    if (output) {
        world.pods[0].output(pop[better_sol_id][0].mv[0]);
        world.pods[1].output(pop[better_sol_id][0].mv[1]);
        expected = world;
        expected.play(pop[better_sol_id][0], otherPlayer[0]);
    } else {
        for (int i = 0; i < profondeur; i++) {
            enemyMoves[i] = pop[better_sol_id][i];
        }
    }
}

int laps[NBPOD];
int lcpid[NBPOD];

int main()
{
    laps[0] = 0;
    laps[1] = 0;
    laps[2] = 0;
    laps[3] = 0;
    lcpid[0] = -1;
    lcpid[1] = -1;
    lcpid[2] = -1;
    lcpid[3] = -1;
    int turn = 0;
    int id_player;
    int prof = 8;
    int pop = 10;
    enemyMoves = new Solution[prof];

    cin >> nblaps; cin.ignore();
    cin >> checkpointCount; cin.ignore();
    for (int i = 0; i < checkpointCount; i++) {
        int cx, cy;
        cin >> cx >> cy; cin.ignore();
        circuit.cps.push_back(Checkpoint(i,cx,cy));
    }

    World w;
    for (int i = 0; i < NBPOD; i++) {
        w.pods[i].id = i;
        w.pods[i].boost_available = true;
        w.pods[i].ncpid = 1;
    }
    // game loop
    while (1) {
        turn++;
        if (turn != 1)
            w = expected;
        for (int i = 0; i < NBPOD; i++) {
            int x,y,vx,vy,angle,nextCheckpointId;
            cin >> x >> y >> vx >> vy >> angle >> nextCheckpointId; cin.ignore();
            cerr << i << " lap " << w.pods[i].lap << " cp " << nextCheckpointId << " score " << w.pods[i].score() << endl;
            //assert(turn == 1 || angle != -1);
            if (nextCheckpointId < 0) {
                cout << "0 0 0\n";
                cout << "0 0 0\n";
                return 0;
            }
            w.pods[i].x = x;
            w.pods[i].y = y;
            w.pods[i].vx = vx;
            w.pods[i].vy = vy;
            if (angle == -1) {
                w.pods[i].angle = w.pods[i].getAngle(circuit.cp(1));
            } else {
                if (i >= 2) {
                    w.pods[i].angle = angle;
                }
                else {
                    #ifndef DOOM_BOT
                    if (abs(w.pods[i].angle - angle) > 0.5) {
                        cout << "0 0 0 angle error\n";
                        cout << "0 0 0 angle error\n";
                        continue;
                    }
                    #endif
                    #ifdef DOOM_BOT
                    w.pods[i].angle = angle;
                    #endif // DOOM_BOT
                }
            }
            if (lcpid[i] == 0 && nextCheckpointId == 1)
                laps[i]++;
            w.pods[i].ncpid = nextCheckpointId;
            w.pods[i].lap = laps[i];

            lcpid[i] = nextCheckpointId;
        }
        float computing_time;
        if (turn == 1) {
            computing_time = 0.800;
            if (w.pods[0].distance(w.pods[1]) > 2000)
                id_player = 1;
            else
                id_player = 0;
        }
        else
            computing_time = 0.068;

        #ifdef DEBUG_SIMU_MODE
        if (turn != 1)
            w.compare(expected);
        if (id_player == 1) {
            cout << "0 0 0\n";
            cout << "0 0 0\n";
            continue;
        }
        else {
            Solution emptyMoves[prof];
            genetic(prof,5,computing_time,w,emptyMoves,true);
        }
        #else

        #ifdef DOOM_BOT
        Move mv1 = convert_to_move(w.pods[0],circuit.cp(w.pods[0].ncpid));
        Move mv2 = convert_to_move(w.pods[1],circuit.cp(w.pods[1].ncpid));
        cerr << "pod 0 wants to go to " << round(circuit.cp(w.pods[0].ncpid).x) << " " << round(circuit.cp(w.pods[0].ncpid).y) << endl;
        cerr << "mv1 : " << mv1.g1 << " " << mv1.g2 << " " << mv1.g3 << endl;
        cerr << "translates to angle/power " << w.pods[0].get_new_angle(mv1.g1) << "/" << w.pods[0].get_new_power(mv1.g2) << endl;
        w.pods[0].output(mv1);
        w.pods[1].output(mv2);
        continue;
        #endif

        Solution emptyMoves[prof];
        Solution otherPlayer[prof];
        World doom_bot = w;
        for (int i = 0; i < prof; i++) {
            Point d1, d2;
            if (doom_bot.pods[0].score() > doom_bot.pods[1].score()) {
                d1 = circuit.cp(doom_bot.pods[0].ncpid);
                d2 = (doom_bot.pods[2].score() > doom_bot.pods[3].score())?doom_bot.pods[2]:doom_bot.pods[3];
            } else {
                d1 = (doom_bot.pods[2].score() > doom_bot.pods[3].score())?doom_bot.pods[2]:doom_bot.pods[3];
                d2 = circuit.cp(doom_bot.pods[1].ncpid);
            }
            otherPlayer[i].mv[0] = convert_to_move(doom_bot.pods[0],d1);
            otherPlayer[i].mv[1] = convert_to_move(doom_bot.pods[1],d2);
            doom_bot.play(otherPlayer[i], emptyMoves[i]);
        }
        #ifdef PROFILING
        computing_time = 10;
        #endif
        World reversedWorld = w;
        Pod tmp = w.pods[0];
        reversedWorld.pods[0] = w.pods[2];
        reversedWorld.pods[2] = tmp;
        tmp = w.pods[1];
        reversedWorld.pods[1] = w.pods[3];
        reversedWorld.pods[3] = tmp;
        genetic(prof,pop,0.010,reversedWorld,otherPlayer,false);
        for (int i = 0; i < prof; i++) {
            otherPlayer[i] = enemyMoves[i];
        }
        genetic(prof,pop,computing_time-0.013,w,otherPlayer,true);
        #endif
    }
}

#endif
