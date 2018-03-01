import math

from envs.csb import util


class BaseFeature:
    def to_representation(self):
        raise NotImplementedError('Abstract Method')


class Distance(BaseFeature):
    def __init__(self, distance):
        assert distance >= 0
        self.distance = distance

    def to_representation(self):
        return self.distance / 1000


class Angle(BaseFeature):
    def __init__(self, angle):
        self.angle = angle

    def to_representation(self):
        return self.angle / 360.0


class Pos(BaseFeature):
    def __init__(self, pos):
        self.pos = pos

    def to_representation(self):
        return self.pos / 1000


class Speed(BaseFeature):
    def __init__(self, speed):
        # assert speed >= 0
        self.speed = speed

    def to_representation(self):
        return self.speed / 1000


class PassedCheckpoints(BaseFeature):
    def __init__(self, passed_checkpoints):
        assert passed_checkpoints >= 0
        self.passed_checkpoints = passed_checkpoints

    def to_representation(self):
        return self.passed_checkpoints


class TotalCheckpoints(BaseFeature):
    def __init__(self, total_checkpoints):
        assert total_checkpoints >= 0
        self.total_checkpoints = total_checkpoints

    def to_representation(self):
        return self.total_checkpoints


class ShieldTimer(BaseFeature):
    def __init__(self, shield_timer):
        assert shield_timer <= 4
        if shield_timer < 0:
            shield_timer = 0
        self.shield_timer = shield_timer

    def to_representation(self):
        return self.shield_timer / 4


class Timeout(BaseFeature):
    def __init__(self, timeout):
        assert timeout <= util.TIMEOUT
        if timeout < 0:
            timeout = 0
        self.timeout = timeout

    def to_representation(self):
        return self.timeout / util.TIMEOUT


class BoostAvailable(BaseFeature):
    def __init__(self, boost_available):
        assert boost_available is True or boost_available is False
        self.boost_available = boost_available

    def to_representation(self):
        return float(self.boost_available)


class CompositeFeature(BaseFeature):
    def __init__(self, features):
        self.features = features

    def to_representation(self):
        representation = []
        for feature in self.features:
            r = feature.to_representation()
            if isinstance(r, list):
                for sub_r in r:
                    representation.append(sub_r)
            else:
                representation.append(r)
        return representation


class RelativeCoordinates(CompositeFeature):
    def __init__(self, source, target):
        super().__init__(features=[
            Distance(source.distance(target)),
            Angle(source.getAngle(target)),
        ])


class AbsoluteSpeed(CompositeFeature):
    def __init__(self, pod):
        super().__init__(features=[
            Angle(math.atan2(pod.vy, pod.vx)),
            Speed(math.sqrt(pod.vx**2 + pod.vy**2)),
        ])


class RelativeOrientation(CompositeFeature):
    def __init__(self, source, target):
        super().__init__(features=[
            Angle(source.diffAngle(target)),
        ])


class RelativeInformation(CompositeFeature):
    def __init__(self, source, target):
        super().__init__(features=[
            AbsoluteSpeed(target),
            RelativeCoordinates(source, target),
            RelativeOrientation(source, target),
        ])


class PodFeature(CompositeFeature):
    def __init__(self, world, pod, allied_pod, best_enemy_pod, second_enemy_pod):
        super().__init__(features=[
            AbsoluteSpeed(pod),
            Angle(pod.angle),
            # BoostAvailable(pod.boost_available),
            # Timeout(pod.timeout),
            # ShieldTimer(pod.shield),
            # PassedCheckpoints(pod.nbChecked()),
            # RelativeInformation(pod, allied_pod),
            # RelativeInformation(pod, best_enemy_pod),
            # RelativeInformation(pod, second_enemy_pod),
            RelativeCoordinates(pod, pod.next_checkpoint(world, 0)),
            # RelativeCoordinates(pod, pod.next_checkpoint(world, 1)),
            # RelativeCoordinates(pod, pod.next_checkpoint(world, 2)),
            # RelativeCoordinates(pod, pod.next_checkpoint(world, 3)),
            # RelativeCoordinates(pod, pod.next_checkpoint(world, 4)),
            # RelativeCoordinates(pod, pod.next_checkpoint(world, 5)),
        ])


class Observation(CompositeFeature):
    def __init__(self, world):
        features = [TotalCheckpoints(world.circuit.nbcp() * world.nblaps)]
        for pod in world.pods:
            features += [
                Pos(pod.x),
                Pos(pod.y),
                Speed(pod.vx),
                Speed(pod.vy),
                Angle(pod.angle),
                BoostAvailable(pod.boost_available),
                Timeout(pod.timeout),
                ShieldTimer(pod.shield),
                PassedCheckpoints(pod.nbChecked()),
            ]
            for i in range(5):
                next_cp = pod.next_checkpoint(world, i)
                features += [
                    Speed(next_cp.x - pod.x),
                    Speed(next_cp.y - pod.y),
                ]
        super().__init__(features=features)
