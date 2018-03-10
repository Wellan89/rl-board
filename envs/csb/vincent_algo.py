import math
import copy
import sys

from envs.csb.util import LIN

FRICTION_COEFF = 0.15
MAX_ROTATE_ANGLE = 18.0


def restoreAngle(angle):
    while angle >= 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def clamp(val, minVal, maxVal):
    return max(min(val, maxVal), minVal)


POD_1 = 0
POD_2 = 1
PODS_COUNT = 2

ME = 0
OPPONENT = 1
PLAYERS_COUNT = 2

FIRST = 0
SECOND = 1


class Pos:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def distToSq(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy

    def distTo(self, other):
        return math.sqrt(self.distToSq(other))

    def normSq(self):
        return self.x * self.x + self.y * self.y

    def norm(self):
        return math.sqrt(self.normSq())

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self

    def __imul__(self, val):
        self.x *= val
        self.y *= val
        return self

    def __add__(self, other):
        return Pos(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Pos(self.x - other.x, self.y - other.y)

    def __mul__(self, val):
        return Pos(self.x * val, self.y * val)

    def __truediv__(self, val):
        return Pos(self.x / val, self.y / val)

    def normalize(self):
        nrm = self.norm()
        if nrm != 0.0:
            self.x /= nrm
            self.y /= nrm

    def normalized(self):
        res = Pos(self.x, self.y)
        res.normalize()
        return res

    def rotate(self, angle):
        angle = angle * math.pi / 180.0
        cs = math.cos(angle)
        sn = math.sin(angle)
        lastX = self.x
        self.x = self.x * cs - self.y * sn
        self.y = lastX * sn + self.y * cs

    def rotated(self, angle):
        res = Pos(self.x, self.y)
        res.rotate(angle)
        return res

    def getAngleDeg(self):
        return math.atan2(self.y, self.x) * 180.0 / math.pi


Vec = Pos


class GameConstants:

    def __init__(self, lapsCount, checkpoints, turn):
        self.lapsCount = lapsCount
        self.checkpoints = checkpoints
        self.turn = turn


gameCsts = GameConstants(0, 0, 0)


class PodCommand:

    def __init__(self, target, thrust, shield=False):
        self.target = target
        self.thrust = thrust
        self.shield = shield


class PodState:
    def __init__(self, pos, speed, angle):
        self.pos = pos
        self.speed = speed
        self.angle = angle


class Pod:

    def __init__(self):
        self.state = PodState(Pos(), Vec(), 0)
        self.lastState = PodState(Pos(), Vec(), 0)

        self.lap = 0
        self.nextCheckpointIdx = 1
        self.checkpointsPassed = 0
        self.rank = FIRST

    def getNextCheckpoint(self, next=1):
        idx = (next + self.nextCheckpointIdx - 1)
        idx %= len(gameCsts.checkpoints)
        return gameCsts.checkpoints[idx]

    def estimateLastCmd(self):
        F = self.state.speed / (1.0 - FRICTION_COEFF) - self.lastState.speed
        Fnorm = F.norm()
        return PodCommand(
            target=self.lastState.pos + F * (10000.0 / Fnorm) if Fnorm != 0 else self.state.pos,
            thrust=clamp(Fnorm, 0.0, 200.0)
        )

    def estimateNextCmd(self):
        cmd = self.estimateLastCmd()
        cmd.target = cmd.target - self.lastState.pos + self.state.pos
        return cmd

    def expectedNextState(self, cmd):
        expected = copy.deepcopy(self.state)
        toTarget = cmd.target - expected.pos
        deltaAngle = clamp(
            restoreAngle(toTarget.getAngleDeg() - expected.angle),
            -MAX_ROTATE_ANGLE, MAX_ROTATE_ANGLE)
        expected.angle += deltaAngle
        if cmd.thrust > 0.0:
            dir = Vec(1.0, 0.0).rotated(expected.angle)
            expected.speed += dir * cmd.thrust

        expected.pos += expected.speed
        expected.speed *= (1.0 - FRICTION_COEFF)
        return expected

    def expectHitWith(self, other, myCmd, otherCmd):
        expectedPos = self.expectedNextState(myCmd).pos
        otherExpectedPos = other.expectedNextState(otherCmd).pos
        collisionRadius = 780.0
        return (expectedPos - otherExpectedPos).normSq() < (collisionRadius * collisionRadius)

    def shouldActivateShieldAgainst(self, other, myCmd, otherCmd):

        if not self.expectHitWith(other, myCmd, otherCmd):
            return False

        nextCheckpoint = self.getNextCheckpoint()
        toTarget = (nextCheckpoint - self.state.pos)

        minBackwardAngle = 60.0
        targetAngle = toTarget.getAngleDeg()
        hitAngle = (other.state.pos - self.state.pos).getAngleDeg()
        angleDiff = abs(restoreAngle(targetAngle - hitAngle - 180.0))
        if angleDiff < minBackwardAngle:
            return False

        minCheckpointRadius = 1500.0
        if toTarget.normSq() < minCheckpointRadius * minCheckpointRadius:
            return True

        minDiffSpeed = 250.0
        speedDiffNormSq = (self.state.speed - other.state.speed).normSq()
        if speedDiffNormSq < minDiffSpeed * minDiffSpeed:
            return False
        return True

    def cmdGoTo(self, target, nextTarget):

        targetSpeed = 1200.0
        accelAngle = 45.0
        minSpeed = 1.0
        maxThrust = 200.0

        startRotating = False
        thrustTowardNext = False

        if nextTarget:
            minSpeedAtTarget = 100.0
            maxDistAtTarget = 500.0
            maxPredictionTurns = 5

            future = copy.deepcopy(self)
            for i in range(maxPredictionTurns):
                toTarget = (nextTarget - future.state.pos).normalized()
                force = toTarget * targetSpeed + future.state.speed * (FRICTION_COEFF - 1.0)

                futureCmd = PodCommand(
                    target=future.state.pos + force.normalized() * 10000.0,
                    thrust=0.0,
                )

                future.state = future.expectedNextState(futureCmd)

                if ((future.state.pos - target).normSq() < maxDistAtTarget * maxDistAtTarget and
                        future.state.speed.normSq() > minSpeedAtTarget * minSpeedAtTarget):

                    startRotating = True
                    break

            if startRotating:

                deltaAngle = restoreAngle((nextTarget - self.state.pos).getAngleDeg() - self.state.angle)
                if (abs(deltaAngle) < MAX_ROTATE_ANGLE):

                    future = copy.deepcopy(self)
                    for i in range(maxPredictionTurns):

                        toTarget = (nextTarget - future.state.pos).normalized()
                        force = toTarget * targetSpeed + future.state.speed * (FRICTION_COEFF - 1.0)

                        futureCmd = PodCommand(
                            target=future.state.pos + force.normalized() * 10000.0,
                            thrust=min(maxThrust, force.norm()),
                        )
                        future.state = future.expectedNextState(futureCmd)

                        if ((future.state.pos - target).normSq() < maxDistAtTarget * maxDistAtTarget and
                                future.state.speed.normSq() > minSpeedAtTarget * minSpeedAtTarget):
                            thrustTowardNext = True
                            break

        tgt = nextTarget if startRotating else target
        toTarget = (tgt - self.state.pos).normalized()
        force = toTarget * targetSpeed + self.state.speed * (FRICTION_COEFF - 1.0)

        cmd_target = self.state.pos + force.normalized() * 10000.0
        deltaAngle = abs(restoreAngle((cmd_target - self.state.pos).getAngleDeg() - self.state.angle))
        if ((startRotating and not thrustTowardNext) or
                (self.state.speed.norm() >= minSpeed and deltaAngle >= accelAngle)):
            cmd_thrust = 0.0
        else:
            cmd_thrust = min(maxThrust, force.norm())
        return PodCommand(
            target=cmd_target,
            thrust=cmd_thrust
        )

    def read(self, x, y, vx, vy, angle, nextCheckpointIdx):
        self.lastState = self.state
        self.state.pos.x = x
        self.state.pos.y = y
        self.state.speed.x = vx
        self.state.speed.y = vy
        self.state.angle = restoreAngle(angle)

        if nextCheckpointIdx != self.nextCheckpointIdx:
            self.checkpointsPassed += 1
            if nextCheckpointIdx < self.nextCheckpointIdx:
                self.lap += 1

        if nextCheckpointIdx < 0 or nextCheckpointIdx >= len(gameCsts.checkpoints):
            nextCheckpointIdx = 0
        self.nextCheckpointIdx = nextCheckpointIdx


class Player:

    def __init__(self):
        self.pods = [Pod(), Pod()]

    def updatePodsRank(self):

        if (self.pods[POD_1].checkpointsPassed > self.pods[POD_2].checkpointsPassed):
            self.pods[POD_1].rank = FIRST
            self.pods[POD_2].rank = SECOND
        elif (self.pods[POD_1].checkpointsPassed < self.pods[POD_2].checkpointsPassed):
            self.pods[POD_1].rank = SECOND
            self.pods[POD_2].rank = FIRST
        else:
            if (self.pods[POD_1].state.pos.distToSq(self.pods[POD_1].getNextCheckpoint()) <=
                    self.pods[POD_2].state.pos.distToSq(self.pods[POD_2].getNextCheckpoint())):
                self.pods[POD_1].rank = FIRST
                self.pods[POD_2].rank = SECOND
            else:
                self.pods[POD_1].rank = SECOND
                self.pods[POD_2].rank = FIRST


class VincentSalimInterface:

    def start(self, salimWorld):
        global gameCsts
        gameCsts = GameConstants(
            salimWorld.nblaps,
            [Pos(cp.x, cp.y) for cp in salimWorld.circuit.cps],
            0,
        )
        self.players = [Player(), Player()]

    def feed(self, salimWorld):
        gameCsts.turn += 1
        current_pod_id = 0
        for player in self.players:
            for pod in player.pods:
                pod.read(
                    x=salimWorld.pods[current_pod_id].x,
                    y=salimWorld.pods[current_pod_id].y,
                    vx=salimWorld.pods[current_pod_id].vx,
                    vy=salimWorld.pods[current_pod_id].vy,
                    angle=restoreAngle(salimWorld.pods[current_pod_id].angle),
                    nextCheckpointIdx=salimWorld.pods[current_pod_id].ncpid,
                )
                current_pod_id += 1

    def get_moves(self, salimWorld, playerId):
        cmd = self.generateCommand(playerId)
        return [
            pod.genes_from_vincent_command(cmd[i])
            for i, pod in enumerate(salimWorld.pods[playerId*2:playerId*2+2])
        ]

    def generateCommand(self, playerId):

        cmd = [None, None]

        for i in range(PODS_COUNT):

            pod = self.players[playerId].pods[i]
            target = self.players[playerId].pods[i].getNextCheckpoint()
            nextTarget = self.players[playerId].pods[i].getNextCheckpoint(2)

            cmd[i] = pod.cmdGoTo(target, nextTarget)

            if i == POD_2 and gameCsts.turn < 3:
                cmd[i].thrust = 0

            if gameCsts.turn > 2:
                for enemy_pod in self.players[1-playerId].pods:
                    if pod.shouldActivateShieldAgainst(enemy_pod, cmd[i], enemy_pod.estimateNextCmd()):
                        cmd[i].shield = True

        return cmd
