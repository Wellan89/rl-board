from collections import namedtuple

STARTING_GOLD = 100
QUEEN_SPEED = 60
TOWER_HP_INITIAL = 200
TOWER_HP_INCREMENT = 100
TOWER_HP_MAXIMUM = 800
TOWER_CREEP_DAMAGE_MIN = 3
TOWER_CREEP_DAMAGE_CLIMB_DISTANCE = 200
TOWER_QUEEN_DAMAGE_MIN = 1
TOWER_QUEEN_DAMAGE_CLIMB_DISTANCE = 200
TOWER_MELT_RATE = 4
TOWER_COVERAGE_PER_HP = 1000
GIANT_BUST_RATE = 80
OBSTACLE_GAP = 90
OBSTACLE_RADIUS_RANGE = range(60, 90)
OBSTACLE_GOLD_RANGE = range(200, 250)
OBSTACLE_MINE_BASESIZE_RANGE = range(1, 3)
OBSTACLE_GOLD_INCREASE = 50
OBSTACLE_GOLD_INCREASE_DISTANCE_1 = 500
OBSTACLE_GOLD_INCREASE_DISTANCE_2 = 200
OBSTACLE_PAIRS = range(6, 12)

KNIGHT_DAMAGE = 1
ARCHER_DAMAGE = 2
ARCHER_DAMAGE_TO_GIANTS = 10

QUEEN_RADIUS = 30
QUEEN_MASS = 10000
QUEEN_HP = range(5, 20)
QUEEN_HP_MULT = 5
QUEEN_VISION = 300

WORLD_WIDTH = 1920
WORLD_HEIGHT = 1000

TOUCHING_DELTA = 5
WOOD_FIXED_INCOME = 10

LEAGUES_towers = True
LEAGUES_giants = True
LEAGUES_mines = True
LEAGUES_fixedIncome = None
LEAGUES_obstacles = 12
LEAGUES_queenHp = 100

CreepType = namedtuple('CreepType', [
    'count', 'cost', 'speed', 'range', 'radius', 'mass', 'hp', 'buildTime', 'assetName'
])

KNIGHT = CreepType(4, 80, 100, 0, 20, 400, 30, 5, "Unite_Fantassin")
ARCHER = CreepType(100, 75, 200, 25, 900, 45, 8, "Unite_Archer")
GIANT = CreepType(1, 140, 50, 0, 40, 2000, 200, 10, "Unite_Siege")


NO_STRUCTURE = -1
TOWER = ??
BARRACKS = 2
MINE = ??
