#ifndef DEF_SMASH_THE_CODE_H
#define DEF_SMASH_THE_CODE_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

using namespace std;

#define NB_LINES            12
#define NB_COLUMNS          6
#define NB_BLOCKS_KNOWN     8

#define NB_COLORS           5
#define EMPTY_CELL          5
#define SKULL_CELL          6
#define TOTAL_NB_COLORS     7

struct Pos
{
    int i;
    int j;

    Pos() = default;
    Pos(int i, int j) : i(i), j(j)
    {
    }

    bool operator==(const Pos& other) const
    {
        return (i == other.i && j == other.j);
    }
};

struct Block
{
    char colorA;
    char colorB;

    Block(char colorA, char colorB) : colorA(colorA), colorB(colorB)
    {
    }
};

struct Command
{
    int column = 0;
    int rotation = 0;

    Command() = default;
    Command(int column, int rotation) : column(column), rotation(rotation)
    { }

    bool isValid() const
    {
        if (column < 0 || column >= NB_COLUMNS)
            return false;

        if (rotation == 0 && column >= NB_COLUMNS - 1)
            return false;

        if (rotation == 3 && column < 1)
            return false;

        return true;
    }

    bool operator==(const Command& other) const {
        return (column == other.column && rotation == other.rotation);
    }
};
const Command allCommands[22] = {
    Command(2, 0), Command(2, 1), Command(2, 2), Command(2, 3),
    Command(3, 0), Command(3, 1), Command(3, 2), Command(3, 3),
    Command(4, 0), Command(4, 1), Command(4, 2), Command(4, 3),
    Command(5, 1), Command(5, 2), Command(5, 3),
    Command(0, 0), Command(0, 1), Command(0, 3),
    Command(1, 0), Command(1, 1), Command(1, 2), Command(1, 3)
};
int getCommandIdx(const Command& cmd) {
    return distance(allCommands, find(allCommands, allCommands + 22, cmd));
}

class Grid
{
protected:
    char grid[NB_LINES][NB_COLUMNS];

public:
    int score = 0;
    float skullPoints = 0.0f;

    Grid() {
        for (int i = 0; i < NB_LINES; i++)
            for (int j = 0; j < NB_COLUMNS; j++)
                grid[i][j] = EMPTY_CELL;
    }

    char& operator[](const Pos& p)
    {
        return grid[p.i][p.j];
    }
    char operator[](const Pos& p) const
    {
        return grid[p.i][p.j];
    }

    float applyCommand(const Block& block, const Command& cmd, bool bruteforceReward)
    {
        switch (cmd.rotation)
        {
        case 0:
            return applyHorizontal(block, cmd.column, bruteforceReward);
        case 1:
            return applyVertical(block, cmd.column, bruteforceReward);
        case 2:
            return applyHorizontal(Block(block.colorB, block.colorA), cmd.column - 1, bruteforceReward);
        default: //case 3:
            return applyVertical(Block(block.colorB, block.colorA), cmd.column, bruteforceReward);
        }
    }

    bool canApplyCommand(Command cmd) const
    {
        if (cmd.rotation == 1 || cmd.rotation == 3)
            return canPoseBlockVertical(cmd.column);
        else if (cmd.rotation == 2)
            return canPoseBlockHorizontal(cmd.column - 1);
        else
            return canPoseBlockHorizontal(cmd.column);
    }

    int getNbFreeCells() const
    {
        int nbFreeCells = 0;
        for (int i = 0; i < NB_LINES; i++)
            for (int j = 0; j < NB_COLUMNS; j++)
                if (grid[i][j] == EMPTY_CELL)
                    nbFreeCells++;
        return nbFreeCells;
    }

    vector<Pos> applyCommandGetPos(const Block& block, const Command& cmd)
    {
        switch (cmd.rotation)
        {
        case 0:
            return applyHorizontalPositions(block, cmd.column);
        case 1:
            return applyVerticalPositions(block, cmd.column);
        case 2:
            return applyHorizontalPositions(Block(block.colorB, block.colorA), cmd.column - 1);
        default: //case 3:
            return applyVerticalPositions(Block(block.colorB, block.colorA), cmd.column);
        }
    }

    std::pair<int, int> detectAndDestroyGroup(const Pos& startPos, vector<Pos>& positions)
    {
        Grid old(*this);

        const int color = grid[startPos.i][startPos.j];
        if (color == SKULL_CELL || color == EMPTY_CELL)
            return std::pair<int, int>(0, 0);

        int nbBlocks = 0;
        vector<Pos> group;
        group.push_back(startPos);
        while (!group.empty())
        {
            Pos pos = group.back();
            group.pop_back();

            if (grid[pos.i][pos.j] == SKULL_CELL)
            {
                grid[pos.i][pos.j] = EMPTY_CELL;
                continue;
            }
            if (grid[pos.i][pos.j] != color)
                continue;

            grid[pos.i][pos.j] = EMPTY_CELL;
            nbBlocks++;

            auto it = find(positions.begin(), positions.end(), pos);
            if (it != positions.end())
                positions.erase(it);

            if (pos.i > 0)
                group.emplace_back(pos.i - 1, pos.j);
            if (pos.i < NB_LINES - 1)
                group.emplace_back(pos.i + 1, pos.j);
            if (pos.j > 0)
                group.emplace_back(pos.i, pos.j - 1);
            if (pos.j < NB_COLUMNS - 1)
                group.emplace_back(pos.i, pos.j + 1);
        }

        int realNbBlocks = nbBlocks;

        if (nbBlocks < 4)
        {
            (*this) = old;
            nbBlocks = 0;
        }

        return std::pair<int, int>(nbBlocks, realNbBlocks);
    }

    void sendSkulls(int nb_skulls) {
        for (int j = 0; j < NB_COLUMNS; j++) {
            int column_nb_skulls_rem = nb_skulls;
            for (int i = NB_LINES - 1; i >= 0; i--) {
                if (grid[i][j] == EMPTY_CELL) {
                    grid[i][j] = SKULL_CELL;
                    --column_nb_skulls_rem;
                    if (column_nb_skulls_rem <= 0)
                        break;
                }
            }
        }
    }

protected:
    void applyGravity(vector<Pos>& outModifiedPositions)
    {
        for (int j = 0; j < NB_COLUMNS; j++)
        {
            int firstEmptyCell = NB_LINES - 1;
            for (int i = NB_LINES - 1; i >= 0; i--)
            {
                if (grid[i][j] != EMPTY_CELL)
                {
                    if (firstEmptyCell != i)
                    {
                        grid[firstEmptyCell][j] = grid[i][j];
                        grid[i][j] = EMPTY_CELL;
                        outModifiedPositions.emplace_back(firstEmptyCell, j);
                    }
                    firstEmptyCell--;
                }
            }
        }
    }
    float update(vector<Pos>&& positions, bool bruteforceReward)
    {
        int turnScore = 0;
        int CP = 0;
        int nbCombos = 0;
        bool first = true;
        int adjacencyScore = 0;
        while (true)
        {
            int B = 0;
            int nbCouleurs = -1;
            int GB = 0;
            while (!positions.empty())
            {
                Pos pos = positions.back();
                positions.pop_back();

                std::pair<int, int> destroyRes = detectAndDestroyGroup(pos, positions);
                int nbBlocks = destroyRes.first;
                int realGroupSize = destroyRes.second;
                if (nbBlocks >= 4)
                {
                    B += nbBlocks;
                    nbCouleurs++;
                    GB += (nbBlocks > 10 ? 8 : (nbBlocks - 4));
                }

                if (first)
                    adjacencyScore += realGroupSize;
            }
            first = false;
            if (B == 0)
                break;

            int CB = (nbCouleurs <= 0 ? 0 : (1 << (nbCouleurs - 1)));
            int sumC = min(max(CP + CB + GB, 1), 999);
            turnScore += (10 * B) * sumC;
            nbCombos++;

            if (CP == 0)
                CP = 8;
            else
                CP *= 2;

            applyGravity(positions);
        }
        score += turnScore;
        skullPoints += turnScore / 70.0f;

        if (bruteforceReward) {
            return float(1 << max(2 * nbCombos - 1, 1)) + adjacencyScore * 0.1f;
        } else {
            //return (nbCombos > 0 ? pow(4, nbCombos - 1) * 0.1f : 0.0f);
            return turnScore;
        }
    }

    vector<Pos> applyVerticalPositions(const Block& block, int column)
    {
        vector<Pos> positions;
        for (int i = NB_LINES - 1; i >= 0; i--)
        {
            if (grid[i][column] == EMPTY_CELL)
            {
                grid[i][column] = block.colorA;
                grid[i - 1][column] = block.colorB;
                positions.emplace_back(i, column);
                positions.emplace_back(i - 1, column);
                break;
            }
        }
        return positions;
    }
    vector<Pos> applyHorizontalPositions(const Block& block, int column)
    {
        vector<Pos> positions;
        for (int i = NB_LINES - 1; i >= 0; i--)
        {
            if (grid[i][column] == EMPTY_CELL)
            {
                grid[i][column] = block.colorA;
                positions.emplace_back(i, column);
                break;
            }
        }
        for (int i = NB_LINES - 1; i >= 0; i--)
        {
            if (grid[i][column + 1] == EMPTY_CELL)
            {
                grid[i][column + 1] = block.colorB;
                positions.emplace_back(i, column + 1);
                break;
            }
        }
        return positions;
    }
    float applyVertical(const Block& block, int column, bool bruteforceReward)
    {
        return update(applyVerticalPositions(block, column), bruteforceReward);
    }
    float applyHorizontal(const Block& block, int column, bool bruteforceReward)
    {
        return update(applyHorizontalPositions(block, column), bruteforceReward);
    }

    bool canPoseBlockVertical(int column) const
    {
        return (grid[0][column] == EMPTY_CELL && grid[1][column] == EMPTY_CELL);
    }
    bool canPoseBlockHorizontal(int column) const
    {
        return (grid[0][column] == EMPTY_CELL && grid[0][column + 1] == EMPTY_CELL);
    }
};

class GameState
{
public:
    Grid myGrid;
    Grid oppGrid;
    vector<Block> nextBlocks;

protected:
    static void sendSkulls(Grid& sourceGrid, Grid& distGrid) {
        int sourceSkullLines = int(sourceGrid.skullPoints / 6.0f);
        if (sourceSkullLines > 0) {
            sourceGrid.skullPoints -= sourceSkullLines * 6.0f;
            distGrid.sendSkulls(sourceSkullLines);
        }
    }

public:
    GameState() {
        for (int i = 0; i < NB_BLOCKS_KNOWN; i++)
            nextBlocks.emplace_back(rand() % NB_COLORS, rand() % NB_COLORS);
    }

    GameState reversed() const {
        GameState ret(*this);
        ret.myGrid = oppGrid;
        ret.oppGrid = myGrid;
        return ret;
    }

    array<float, NB_BLOCKS_KNOWN> estimateMultCoeff() const
    {
        int myNbFreeCells = myGrid.getNbFreeCells();
        int oppNbFreeCells = oppGrid.getNbFreeCells();
        float gamma;
        if (myNbFreeCells <= 22)
            gamma = 0.01f;
        else if (myNbFreeCells <= 44 || oppNbFreeCells <= 33)
            gamma = 0.5f;
        else
            gamma = 0.99f;

        array<float, NB_BLOCKS_KNOWN> multCoeffs;
        multCoeffs[0] = 1.0f;
        for (size_t i = 1; i < multCoeffs.size(); i++)
            multCoeffs[i] = multCoeffs[i - 1] * gamma;
        return multCoeffs;
    }

    float applyMyCommand(Command cmd)
    {
        float reward = myGrid.applyCommand(nextBlocks.back(), cmd, true);
        nextBlocks.pop_back();
        return reward;
    }

    std::pair<float, float> play(Command myCmd, Command oppCmd) {
        float myReward = myGrid.applyCommand(nextBlocks.back(), myCmd, false);
        float oppReward = oppGrid.applyCommand(nextBlocks.back(), oppCmd, false);
        sendSkulls(myGrid, oppGrid);
        sendSkulls(oppGrid, myGrid);
        for (int i = nextBlocks.size() - 1; i >= 0; i--)
            nextBlocks[i + 1] = nextBlocks[i];
        nextBlocks[0] = Block(rand() % NB_COLORS, rand() % NB_COLORS);
        return std::pair<float, float>(myReward, oppReward);
    }
};

template<class It>
float evalScoreRec(const GameState& state, It multCoeffsBegin, It multCoeffsEnd, int& nbSimulations)
{
    float bestScore = -numeric_limits<float>::infinity();
    It multCoeffsNext = multCoeffsBegin + 1;
    for (Command cmd : allCommands)
    {
        if (nbSimulations <= 0)
            break;

        if (!state.myGrid.canApplyCommand(cmd))
            continue;

        GameState nextState = state;
        float score = *multCoeffsBegin * nextState.applyMyCommand(cmd);
        --nbSimulations;

        if (multCoeffsNext != multCoeffsEnd)
            score += evalScoreRec(nextState, multCoeffsNext, multCoeffsEnd, nbSimulations);

        if (score > bestScore)
            bestScore = score;
    }
    return bestScore;
}
template<class It>
Command findBestCommandBF(const GameState& state, It multCoeffsBegin, It multCoeffsEnd, int& nbSimulations)
{
    float bestScore = -numeric_limits<float>::infinity();
    It multCoeffsNext = multCoeffsBegin + 1;
    Command bestCommand;
    for (Command cmd : allCommands)
    {
        // Will be aborted in computeBestCommand
        if (nbSimulations <= 0)
            break;

        if (!state.myGrid.canApplyCommand(cmd))
            continue;

        GameState nextState = state;
        float score = nextState.applyMyCommand(cmd);
        --nbSimulations;

        if (multCoeffsNext != multCoeffsEnd)
            score += evalScoreRec(nextState, multCoeffsNext, multCoeffsEnd, nbSimulations);

        if (score > bestScore)
        {
            bestScore = score;
            bestCommand = cmd;
        }
    }

    return bestCommand;
}

Command computeBestCommand(const GameState& state, int nbSimulations)
{
    const auto multCoeffs = state.estimateMultCoeff();

    Command bestCommand = allCommands[0];
    auto multCoeffsEnd = multCoeffs.begin() + 1;
    while (true)
    {
        Command res = findBestCommandBF(state, multCoeffs.begin(), multCoeffsEnd, nbSimulations);
        if (nbSimulations <= 0 || !state.myGrid.canApplyCommand(res))
            break;  // Search aborted
        bestCommand = res;

        if (multCoeffsEnd == multCoeffs.end())
            break;
        ++multCoeffsEnd;
    }
    return bestCommand;
}

#endif
