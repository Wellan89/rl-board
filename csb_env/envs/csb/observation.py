import math


class BaseFeature:
    def to_representation(self):
        raise NotImplementedError('Abstract Method')


class Distance(BaseFeature):
    def __init__(self, distance):
        assert distance >= 0
        self.distance = distance

    def to_representation(self):
        raise NotImplemented('TODO vincent')


class Angle(BaseFeature):
    def __init__(self, angle):
        assert -math.pi <= angle <= math.pi
        self.angle = angle

    def to_representation(self):
        raise NotImplemented('TODO vincent')


class Speed(BaseFeature):
    def __init__(self, speed):
        assert speed >= 0
        self.speed = speed

    def to_representation(self):
        raise NotImplemented('TODO vincent')


class PassedCheckpoints(BaseFeature):
    def __init__(self, passed_checkpoints):
        assert passed_checkpoints >= 0
        self.passed_checkpoints = passed_checkpoints

    def to_representation(self):
        raise NotImplemented('TODO vincent')


class TotalCheckpoints(BaseFeature):
    def __init__(self, total_checkpoints):
        assert total_checkpoints >= 0
        self.total_checkpoints = total_checkpoints

    def to_representation(self):
        raise NotImplemented('TODO vincent')


class ShieldTimer(BaseFeature):
    def __init__(self, shield_timer):
        assert shield_timer <= 4
        if shield_timer < 0:
            shield_timer = 0
        self.shield_timer = shield_timer

    def to_representation(self):
        raise NotImplemented('TODO vincent')


class Timeout(BaseFeature):
    def __init__(self, timeout):
        assert timeout <= 100
        if timeout < 0:
            timeout = 0
        self.timeout = timeout

    def to_representation(self):
        raise NotImplemented('TODO vincent')


class BoostAvailable(BaseFeature):
    def __init__(self, boost_available):
        assert boost_available is True or boost_available is False
        self.boost_available = boost_available

    def to_representation(self):
        raise NotImplemented('TODO vincent')


class CompositeFeature:
    def to_representation(self):
        reprensentation = []
        for feature in self.features:
            r = feature.to_representation()
            if isinstance(r, list):
                for sub_r in r:
                    reprensentation.append(sub_r)
            else:
                reprensentation.append(r)
        return reprensentation


class RelativeCoordinates(CompositeFeature):
    def __init__(self, source, target):
        self.features = [
            Distance(source.distance(target)),
            Angle(0),
        ]


class RelativeSpeed(CompositeFeature):
    def __init__(self, source, target):
        self.features = [
            Angle(0),
            Speed(math.sqrt(target.vx**2 + target.vy**2)),
        ]


class RelativeInformation(CompositeFeature):
    def __init__(self, source, target):
        self.features = [
            RelativeSpeed(source, target),
            RelativeCoordinates(source, target),
        ]


class PodFeature(CompositeFeature):
    def __init__(self, world, pod, allied_pod, best_enemy_pod, second_enemy_pod):
        self.features = [
            RelativeSpeed(pod, pod),
            BoostAvailable(pod.boost_available),
            Timeout(pod.timeout),
            ShieldTimer(pod.shield),
            PassedCheckpoints(pod.nbChecked()),
            RelativeInformation(pod, allied_pod),
            RelativeInformation(pod, best_enemy_pod),
            RelativeInformation(pod, second_enemy_pod),
            RelativeCoordinates(pod, pod.next_checkpoint(world, 0)),
            RelativeCoordinates(pod, pod.next_checkpoint(world, 1)),
            RelativeCoordinates(pod, pod.next_checkpoint(world, 2)),
            RelativeCoordinates(pod, pod.next_checkpoint(world, 3)),
            RelativeCoordinates(pod, pod.next_checkpoint(world, 4)),
            RelativeCoordinates(pod, pod.next_checkpoint(world, 5)),
        ]


class Observation(CompositeFeature):

    def __init__(self, world):

        self.features = [
            TotalCheckpoints(world.circuit.nbcp() * world.nblaps),
            PodFeature(world, world.best_pod(0), world.second_pod(0), world.best_pod(1), world.second_pod(1)),
            PodFeature(world, world.second_pod(0), world.best_pod(0), world.best_pod(1), world.second_pod(1)),
            PodFeature(world, world.best_pod(1), world.second_pod(1), world.best_pod(0), world.second_pod(0)),
            PodFeature(world, world.second_pod(1), world.best_pod(1), world.best_pod(0), world.second_pod(0)),
        ]
