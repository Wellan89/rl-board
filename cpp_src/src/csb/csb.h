#ifndef DEF_CSB_H
#define DEF_CSB_H

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
//#include <assert.h>
#include <cmath>

#define PI 3.14159265358979323846
#define MAX_THRUST 200
#define NBPOD 4

using namespace std;

#define LIN(x, x1, y1, x2, y2) (y1 + (y2-y1)*(x-x1)/(x2-x1))


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
    int nblaps;

    int nbcp() {
        return cps.size();
    }
    Checkpoint& cp(int cpid) {
        return cps[cpid];
    }
    vector<Checkpoint> cps;
};

class Pod : public Unit
{
public:
    Pod(Circuit* _circuit):
        Unit(-1,0.0,0.0,400.0,0.0,0.0), circuit(_circuit) {}
    Circuit* circuit;
    float angle=0.0;
    int ncpid=1;
    int timeout=100;
    int shield=0;
    int lap=0;
    bool boost_available=true;

    void show() {
        cerr << "Pod " << id << " : (" << x << "," << y << ") v(" << vx << "," << vy << ") ang: " << angle << " ncpid:" << ncpid << endl;
    }
    void incrcpid() {
        ncpid++;
        if (ncpid == 1)
            lap++;
        if (ncpid >= circuit->nbcp()) {
            ncpid = 0;
        }
    }
    int nb_checked() {
        int lastCP = ncpid - 1;
        if (lastCP == -1)
            lastCP = circuit->nbcp() - 1;
        return lap * circuit->nbcp() + lastCP;
    }
    float score() {
        return nb_checked() * 50000 - distance(circuit->cp(ncpid));
    }
    float env_score() {
        Checkpoint& current_cp = circuit->cp((ncpid + circuit->nbcp() - 1) % circuit->nbcp());
        Checkpoint& next_cp = circuit->cp(ncpid);
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
        if (this->shield > 0) {
            return;
        }
        float ra = this->angle * PI / 180.0;
        this->vx += cos(ra) * thrust;
        this->vy += sin(ra) * thrust;
    }
    void move(float t) {
        this->x += this->vx*t;
        this->y += this->vy*t;
    }
    void end() {
        this->x = round(this->x);
        this->y = round(this->y);
        this->vx = static_cast<int>(this->vx * 0.85);
        this->vy = static_cast<int>(this->vy * 0.85);
        this->timeout -= 1;
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

class World
{
public:
    World(){}
    Circuit circuit;
    Pod pods[NBPOD] = {Pod(&circuit), Pod(&circuit), Pod(&circuit), Pod(&circuit)};

    Pod save_pods[NBPOD] = {Pod(&circuit), Pod(&circuit), Pod(&circuit), Pod(&circuit)};
    void save() {
        for (int i = 0; i < NBPOD; i++)
            save_pods[i] = pods[i];
    }
    void rollback() {
        for (int i = 0; i < NBPOD; i++)
            pods[i] = save_pods[i];
    }
    World reversed() {
        World reversedWorld(*this);
        reversedWorld.pods[0] = pods[2];
        reversedWorld.pods[1] = pods[3];
        reversedWorld.pods[2] = pods[0];
        reversedWorld.pods[3] = pods[1];
        return reversedWorld;
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

    float eval() {
        if (pods[2].lap == circuit.nblaps || pods[3].lap == circuit.nblaps)
            return -(numeric_limits<float>::max()/2);
        else if (pods[0].lap == circuit.nblaps || pods[1].lap == circuit.nblaps)
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

void genetic(int profondeur, int population, int nb_simulations, World world, Solution* otherPlayer, Solution* output) {
    world.save();

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

    int cur_nb_simulations = 0;
    while (cur_nb_simulations < nb_simulations)
    {
        cur_nb_simulations++;

        world.rollback();
        Solution newsol[profondeur];

        int randid = rand()%population;
        for (int i = 0; i < profondeur; i++) {
            newsol[i] = pop[randid][i];
        }

        for (int i = 0; i < profondeur; i++) {
            newsol[i].mutate(1.0f - float(cur_nb_simulations) / float(nb_simulations));
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
    for (int i = 0; i < profondeur; i++) {
        output[i] = pop[better_sol_id][i];
    }
}

Solution runGenetic(World w, int prof, int pop, int nb_simulations) {
    Solution emptyMoves[prof];
    Solution otherPlayer[prof];
    World doom_bot = w;
    for (int i = 0; i < prof; i++) {
        Point d1, d2;
        if (doom_bot.pods[0].score() > doom_bot.pods[1].score()) {
            d1 = w.circuit.cp(doom_bot.pods[0].ncpid);
            d2 = (doom_bot.pods[2].score() > doom_bot.pods[3].score())?doom_bot.pods[2]:doom_bot.pods[3];
        } else {
            d1 = (doom_bot.pods[2].score() > doom_bot.pods[3].score())?doom_bot.pods[2]:doom_bot.pods[3];
            d2 = w.circuit.cp(doom_bot.pods[1].ncpid);
        }
        otherPlayer[i].mv[0] = convert_to_move(doom_bot.pods[0],d1);
        otherPlayer[i].mv[1] = convert_to_move(doom_bot.pods[1],d2);
        doom_bot.play(otherPlayer[i], emptyMoves[i]);
    }

    Solution enemyMoves[prof];
    genetic(prof, pop, int(nb_simulations * 0.15f), w.reversed(), otherPlayer, enemyMoves);

    Solution output[prof];
    genetic(prof, pop, int(nb_simulations * 0.85f), w, enemyMoves, output);
    return output[0];
}

#endif
