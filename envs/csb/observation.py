from envs.csb import util


def observation(world, pods):
    features = [world.circuit.nbcp() * world.nblaps]
    for pod in pods:
        features += [
            pod.x / 1000.0,
            pod.y / 1000.0,
            pod.vx / 1000.0,
            pod.vy / 1000.0,
            pod.angle / 360.0,
            float(pod.boost_available),
            pod.timeout / 100.0,
            pod.shield / 4.0,
            float(pod.nbChecked()),
        ]
        for i in range(3):
            cp = pod.next_checkpoint(world, i)
            features += [
                cp.x / 1000.0,
                cp.y / 1000.0,
            ]
    return features
