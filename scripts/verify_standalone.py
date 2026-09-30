"""Check that the standalone MJCF matches the installed Python environment."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import tempfile

import mujoco
import numpy as np

from hei_sim import Environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=Path(__file__).resolve().parents[1] / 'scene.xml')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    scene = args.scene.resolve()
    with ExitStack() as stack:
        if os.name == 'nt' and not str(scene).isascii():
            # MuJoCo's native Windows file loader needs an ASCII resource path.
            cache = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='hei_standalone_')))
            shutil.copytree(scene.parent / 'hei_sim/assets', cache / 'hei_sim/assets',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            shutil.copy2(scene, cache / 'scene.xml')
            scene = cache / 'scene.xml'
        standalone = mujoco.MjModel.from_xml_path(str(scene))
    env = Environment()
    try:
        arrays = ('body_pos', 'body_quat', 'body_mass', 'body_inertia',
                  'jnt_pos', 'jnt_axis', 'jnt_range', 'geom_pos', 'geom_quat',
                  'geom_size', 'geom_friction', 'geom_contype', 'geom_conaffinity',
                  'geom_condim', 'actuator_gainprm', 'actuator_biasprm',
                  'actuator_ctrlrange', 'actuator_forcerange', 'cam_pos',
                  'cam_quat', 'cam_fovy', 'cam_bodyid')
        for name in arrays:
            actual, expected = getattr(standalone, name), getattr(env.model, name)
            # MuJoCo's XML export rounds some imported URDF inertial values.
            if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=5e-5, atol=1e-7):
                raise AssertionError(f'Standalone scene differs: {name}')
        options = ('timestep', 'integrator', 'iterations', 'cone', 'impratio',
                   'noslip_iterations', 'enableflags', 'gravity')
        for name in options:
            if not np.allclose(getattr(standalone.opt, name), getattr(env.model.opt, name)):
                raise AssertionError(f'Standalone solver differs: {name}')
        data = mujoco.MjData(standalone)
        mujoco.mj_resetDataKeyframe(standalone, data, standalone.key('home').id)
        for name in ('qpos', 'ctrl'):
            if not np.allclose(getattr(data, name), getattr(env.data, name), rtol=5e-5, atol=1e-7):
                raise AssertionError(f'Home keyframe differs: {name}')
        report = dict(status='passed', compiled_arrays_checked=list(arrays),
                      solver_options_checked=list(options), home_keyframe_match=True,
                      all_meshes_and_textures_loaded=True, mujoco=mujoco.__version__)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
