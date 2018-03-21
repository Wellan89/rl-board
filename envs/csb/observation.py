import time

from envs.csb import circuit
from envs.csb import util


__IMPORT_TIME__ = time.time()


def observation(world, pods, use_timed_features_mask=False):
    hard_features_mask = min((time.time() - __IMPORT_TIME__) / (3600 * 2), 1.0) if use_timed_features_mask else 1.0
    features = [world.circuit.nbcp() * world.nblaps * hard_features_mask]
    for pod in pods:
        pod_features_mask = hard_features_mask if pod.id != 0 else 1.0
        features += [
            pod.x / 1000.0 * pod_features_mask,
            pod.y / 1000.0 * pod_features_mask,
            pod.vx / 1000.0 * pod_features_mask,
            pod.vy / 1000.0 * pod_features_mask,
            pod.angle / 360.0 * pod_features_mask,
        ]
        features += [
            float(pod.boost_available) * hard_features_mask,
            pod.timeout / util.TIMEOUT * hard_features_mask,
            pod.shield / 4.0 * hard_features_mask,
            pod.nbChecked() * hard_features_mask,
        ]
        for i in range(3):
            cp = pod.next_checkpoint(world, i)
            features += [
                cp.x / 1000.0 * pod_features_mask,
                cp.y / 1000.0 * pod_features_mask,
            ]

    for i in range(circuit.MAX_NB_CHECKPOINTS):
        if i < world.circuit.nbcp():
            cp = world.circuit.cp(i)
            features += [
                cp.x / 1000.0 * hard_features_mask,
                cp.y / 1000.0 * hard_features_mask,
            ]
        else:
            features += [
                0.0,
                0.0,
            ]

    return features
