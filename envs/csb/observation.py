import math
import time

from envs.csb import circuit
from envs.csb import util


__IMPORT_TIME__ = time.time()


class BaseFeature:
    def to_representation(self):
        raise NotImplementedError('Abstract Method')


class Distance(BaseFeature):
    def __init__(self, distance):
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


class PassedCheckpoints(BaseFeature):
    def __init__(self, passed_checkpoints):
        self.passed_checkpoints = passed_checkpoints

    def to_representation(self):
        return self.passed_checkpoints


class TotalCheckpoints(BaseFeature):
    def __init__(self, total_checkpoints):
        self.total_checkpoints = total_checkpoints

    def to_representation(self):
        return self.total_checkpoints


class ShieldTimer(BaseFeature):
    def __init__(self, shield_timer):
        if shield_timer < 0:
            shield_timer = 0
        self.shield_timer = shield_timer

    def to_representation(self):
        return self.shield_timer / 4


class Timeout(BaseFeature):
    def __init__(self, timeout):
        if timeout < 0:
            timeout = 0
        self.timeout = timeout

    def to_representation(self):
        return self.timeout / util.TIMEOUT


class BoostAvailable(BaseFeature):
    def __init__(self, boost_available):
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
            Pos(math.sqrt(pod.vx**2 + pod.vy**2)),
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
    def __init__(self, world, variation=None):
        hard_features_mask = min((time.time() - __IMPORT_TIME__) / (3600 * 2), 1.0) if variation == 7 else 1.0
        features = [TotalCheckpoints(world.circuit.nbcp() * world.nblaps * hard_features_mask)]
        for pod in world.pods:
            if pod.id != 0 and (variation == 3 or variation == 5 or variation == 6):
                break

            pod_features_mask = hard_features_mask if pod.id != 0 else 1.0
            features += [
                Pos(pod.x * pod_features_mask),
                Pos(pod.y * pod_features_mask),
                Pos(pod.vx * pod_features_mask),
                Pos(pod.vy * pod_features_mask),
                Angle(pod.angle * pod_features_mask),
            ]
            if variation != 3 and variation != 6:
                features += [
                    BoostAvailable(float(pod.boost_available) * hard_features_mask),
                    Timeout(pod.timeout * hard_features_mask),
                    ShieldTimer(pod.shield * hard_features_mask),
                    PassedCheckpoints(pod.nbChecked() * hard_features_mask),
                ]
            if variation >= 2:
                for i in range(3):
                    cp = pod.next_checkpoint(world, i)
                    features += [
                        Pos(cp.x * pod_features_mask),
                        Pos(cp.y * pod_features_mask),
                    ]

        if variation != 3 and variation != 5 and variation != 6:
            for i in range(circuit.MAX_NB_CHECKPOINTS):
                if i < world.circuit.nbcp():
                    cp = world.circuit.cp(i)
                    features += [
                        Pos(cp.x * hard_features_mask),
                        Pos(cp.y * hard_features_mask),
                    ]
                else:
                    features += [
                        Pos(0.0),
                        Pos(0.0),
                    ]

        super().__init__(features=features)
