"""Compare packaged scene against a fresh independent export of the local scene."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from hei_sim.environment import Environment
from hei_sim.portable import compare, snapshot

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference-dir',type=Path,required=True)
args=parser.parse_args()
export=args.reference_dir.resolve()
spec=importlib.util.spec_from_file_location('_source_reference',export/'load_scene.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source_model,source_data=module.load_scene()
env=Environment()
try:
    compare(json.loads((export/'settings.json').read_text(encoding='utf-8')),snapshot(env))
    names=('body_pos','body_quat','body_mass','body_inertia','jnt_pos','jnt_axis','jnt_range',
        'geom_size','geom_friction','geom_contype','geom_conaffinity','geom_condim',
        'actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange',
        'cam_pos','cam_quat','cam_fovy','cam_bodyid')
    for name in names:
        local,published=getattr(source_model,name),getattr(env.model,name)
        if local.shape!=published.shape or not np.allclose(local,published,rtol=5e-5,atol=1e-7):
            raise AssertionError(f'Packaged scene differs: {name}')
    assert np.allclose(source_data.qpos,env.data.qpos,rtol=5e-5,atol=1e-7)
    print(json.dumps(dict(source_scene_match=True,compiled_arrays_checked=list(names),
        initial_state_match=True,mujoco='3.5.0'),indent=2))
finally:
    env.close()
