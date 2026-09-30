"""Integration checks for contact dynamics, wrist cameras and dataset integrity."""
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np

from .environment import CAMERAS, Environment, ROOT
from .recording import EpisodeWriter, validate_episode, write_png
from .controls import Gamepad, load_config


def main():
    out = ROOT / 'outputs/collection_checks'
    out.mkdir(parents=True, exist_ok=True)
    env = Environment()
    model, data = env.model, env.data
    report = {}
    def put(name, pos, velocity=(0, 0, 0)):
        joint = model.joint('hei_object_' + name + '_free')
        q, v = joint.qposadr[0], joint.dofadr[0]
        data.qpos[q:q+7] = [*pos, 1, 0, 0, 0]
        data.qvel[v:v+6] = [*velocity, 0, 0, 0]
        mujoco.mj_forward(model, data)
    def steps(n):
        for _ in range(n):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
    try:
        put('cube_red', [.95, -.3, 1.15])
        steps(500)
        z = float(data.body('hei_object_cube_red').xpos[2])
        assert abs(z-.82) < .005, z
        report['drop_onto_table_z_m'] = z
        put('cube_red', [.8, -.12, .821], (0, 1, 0))
        put('cube_green', [.8, 0, .821])
        cube_ids = {model.geom('hei_object_cube_red_geom').id, model.geom('hei_object_cube_green_geom').id}
        contact = False
        for _ in range(300):
            mujoco.mj_step(model, data)
            contact |= any({int(c.geom1), int(c.geom2)} == cube_ids for c in data.contact)
        assert contact, 'No object-object collision'
        report['object_object_collision'] = True
        env.reset()
        # Drop a small test cube onto the physical right wrist/finger assembly.
        tcp = data.body('a_right_tcp').xpos.copy()
        put('cube_red', tcp + [0, 0, .16])
        robot_bodies = {i for i in range(model.nbody) if model.body(i).name.startswith('a_right')}
        robot_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] in robot_bodies}
        red = model.geom('hei_object_cube_red_geom').id
        robot_contact = False
        for _ in range(300):
            mujoco.mj_step(model, data)
            robot_contact |= any((c.geom1 == red and c.geom2 in robot_geoms) or
                                  (c.geom2 == red and c.geom1 in robot_geoms) for c in data.contact)
        assert robot_contact, 'No object-robot contact'
        report['robot_object_collision'] = True
        # Pair friction is the max of both surfaces. Change all colliders so the
        # high-friction table does not mask the low-friction object experiment.
        original = model.geom_friction.copy()
        distances = {}
        for label, coefficient in [('low', .001), ('high', .8)]:
            env.reset()
            model.geom_friction[:] = [coefficient, 0, 0]
            put('cube_red', [.9, -.42, .8201], (0, .35, 0))
            steps(150)
            distances[label] = float(data.body('hei_object_cube_red').xpos[1] + .42)
        model.geom_friction[:] = original
        assert distances['low'] > distances['high']+.04, distances
        report['slide_distance_m'] = distances
        env.reset()
        obs = env.observe()
        for name, rgb in obs['images'].items():
            assert rgb.shape == (224,224,3) and rgb.std() > 8
            write_png(out/(name+'.png'), rgb)
        write_png(out/'three_cameras.png', np.concatenate(list(obs['images'].values()), axis=1))
        env.renderer.update_scene(data, camera='overview')
        write_png(out/'overview_not_recorded.png', env.renderer.render().copy())
        assert set(obs['images']) == set(CAMERAS) and 'overview' not in obs['images']
        before_pos = data.cam_xpos.copy()
        before_rot = data.cam_xmat.copy()
        env.target[0] += .1
        env.step()
        front = model.camera('front').id
        overview = model.camera('overview').id
        right = model.camera('right_wrist').id
        assert np.allclose(before_pos[overview], data.cam_xpos[overview])
        assert np.linalg.norm(data.cam_xmat[right]-before_rot[right]) > .02
        left = model.camera('left_wrist').id
        env.target[7] -= .1
        env.step()
        assert np.linalg.norm(data.cam_xmat[left]-before_rot[left]) > .02
        report['wrist_camera_follows_joint'] = True
        assert model.cam_bodyid[front] == model.body('lift_carriage_link').id
        env.target[17] = -.1
        for _ in range(10):
            env.step()
        assert np.linalg.norm(data.cam_xpos[front]-before_pos[front]) > .05
        report['front_camera_on_robot_follows_lift'] = True
        report['overview_excluded_from_observations'] = True
        # Test the input deadman and edge handling independently of a physical
        # person moving sticks. Actual device detection is reported separately.
        pad = Gamepad.__new__(Gamepad)
        pad.config, pad.backend, pad.previous, pad.armed = load_config(), 'winmm', 0, False
        pad.raw = lambda: ({'y':-1,'x':0,'r':0,'u':0}, 16, 65535)
        assert not pad.read()['enabled'], 'Deadman held at startup must not arm'
        pad.raw = lambda: ({'y':-1,'x':0,'r':0,'u':0}, 0, 65535)
        assert not pad.read()['enabled']
        pad.raw = lambda: ({'y':-1,'x':0,'r':0,'u':0}, 16, 65535)
        command = pad.read()
        assert command['enabled'] and command['translation'][0] > .1
        env.reset()
        start_tcp = data.body('a_right_tcp').xpos.copy()
        for _ in range(10):
            env.command('right', command['translation'], command['rotation'], 0)
            env.step()
        assert data.body('a_right_tcp').xpos[0] > start_tcp[0]+.04
        report['gamepad_mapping_to_ik_and_motion'] = True
        report['startup_deadman_interlock'] = True
        env.reset()
        with tempfile.TemporaryDirectory(prefix='hei_record_test_', dir=out) as temp:
            writer = EpisodeWriter(temp, env.metadata(), 'TEST ONLY', test=True)
            incomplete = {**env.observe(), 'images': {'front':obs['images']['front']}}
            try:
                writer.add(incomplete, env.target)
                raise AssertionError('Missing cameras were accepted')
            except ValueError:
                pass
            for _ in range(3):
                obs = env.observe()
                action = env.target.copy()
                env.step(action)
                writer.add(obs, action)
            episode = writer.save()
            meta, rows = validate_episode(episode)
            report['aligned_three_camera_frames'] = len(rows)
            (episode/'left_wrist/000001.png').unlink()
            try:
                validate_episode(episode)
                raise AssertionError('Corrupt episode was accepted')
            except ValueError:
                pass
            report['missing_camera_rejected'] = True
        report['camera_metadata'] = env.metadata()['camera_calibration']
        (out/'checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
