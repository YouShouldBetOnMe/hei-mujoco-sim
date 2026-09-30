"""Portable scene diagnostics/viewer. No training or teleoperation imports."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys
import time
import zlib

import mujoco
import numpy as np

from .environment import CAMERAS, Environment, ROOT, PACKAGE_ROOT


def write_png(path, image):
    """Encode rendered uint8 RGB as PNG, without GUI/image dependencies."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 RGB')
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind+data)&0xffffffff)
    height, width = image.shape[:2]
    rows = b''.join(b'\x00' + row.tobytes() for row in image)
    content = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B',width,height,8,2,0,0,0))
               + chunk(b'IDAT',zlib.compress(rows)) + chunk(b'IEND',b''))
    Path(path).write_bytes(content)


def verify_manifest():
    manifest_path = PACKAGE_ROOT/'ASSET_MANIFEST.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Packaged ASSET_MANIFEST.json is missing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for relative, expected in manifest['sha256'].items():
        path = PACKAGE_ROOT/relative
        if not path.resolve().is_relative_to(PACKAGE_ROOT.resolve()):
            raise ValueError(f'Invalid manifest path: {relative}')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'File differs or is missing: {relative}')
    print(f"Verified {len(manifest['sha256'])} source and asset files.",flush=True)
    return manifest


def snapshot(env):
    """Read actual compiled dimensions/parameters, not hardcoded documentation."""
    m = env.model
    objects = {}
    for obj in env.objects:
        body, geom = m.body(obj.body_name), m.geom(obj.geom_name)
        value = dict(body=obj.body_name, mass_kg=float(body.mass[0]), geom_type=int(geom.type[0]),
            initial_position=m.body_pos[body.id].tolist(), initial_quat_wxyz=m.body_quat[body.id].tolist(),
            friction=geom.friction.tolist(), condim=int(geom.condim[0]),
            contype=int(geom.contype[0]), conaffinity=int(geom.conaffinity[0]))
        if int(geom.type[0]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            value.update(half_size_m=geom.size.tolist(), full_size_m=(2*geom.size).tolist(),
                         volume_m3=float(np.prod(2*geom.size)))
        else:
            mesh = int(geom.dataid[0])
            vertices = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
            rot = np.empty(9)
            mujoco.mju_quat2Mat(rot,geom.quat)
            points = vertices @ rot.reshape(3,3).T + geom.pos
            value.update(mesh_bbox_size_local_m=np.ptp(points,axis=0).tolist(),
                         collision_approximation='single convex hull',
                         bbox_is_not_solid_volume=True)
        objects[obj.body_name] = value
    table = m.geom('hei_table_top')
    robot_collision = np.where(m.geom_contype == 2)[0]
    return dict(physics=env.metadata()['physics'], cameras=env.metadata()['camera_calibration'],
        observation_cameras=list(CAMERAS), excluded_cameras=['overview'], objects=objects,
        timestep=float(m.opt.timestep), integrator=int(m.opt.integrator),
        iterations=int(m.opt.iterations), cone=int(m.opt.cone), impratio=float(m.opt.impratio),
        noslip_iterations=int(m.opt.noslip_iterations),
        multiccd=bool(m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD)),
        table_half_size_m=table.size.tolist(), table_friction=table.friction.tolist(),
        robot_collision_friction=np.unique(m.geom_friction[robot_collision],axis=0).tolist())


def compare(reference, current, path='scene'):
    if isinstance(reference, dict):
        if set(reference) != set(current):
            raise ValueError(f'Different keys at {path}')
        for key in reference:
            compare(reference[key],current[key],path+'.'+key)
    elif isinstance(reference,list):
        if len(reference) != len(current):
            raise ValueError(f'Different length at {path}')
        for i,(r,c) in enumerate(zip(reference,current)):
            compare(r,c,f'{path}[{i}]')
    elif isinstance(reference,(float,int)) and not isinstance(reference,bool):
        if not np.isclose(reference,current,rtol=1e-6,atol=1e-8):
            raise ValueError(f'Different parameter at {path}: {reference} != {current}')
    elif reference != current:
        raise ValueError(f'Different value at {path}: {reference!r} != {current!r}')


def check(env, render=True):
    m,d = env.model,env.data
    report = dict(platform=platform.platform(), architecture=platform.machine(),
                  python=platform.python_version(), mujoco=mujoco.__version__, numpy=np.__version__,
                  render_checked=False)
    reference = PACKAGE_ROOT/'scene_reference.json'
    if reference.is_file() and env.fixed_lift is None:
        compare(json.loads(reference.read_text(encoding='utf-8')),snapshot(env))
        report['scene_matches_exported_reference'] = True
    def place(name, pos, vel=(0,0,0)):
        joint = m.joint('hei_object_'+name+'_free')
        q,v = joint.qposadr[0],joint.dofadr[0]
        d.qpos[q:q+7] = [*pos,1,0,0,0]
        d.qvel[v:v+6] = [*vel,0,0,0]
        mujoco.mj_forward(m,d)
    def step(n):
        for _ in range(n): mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)
    place('cube_red',[.95,-.3,1.15])
    step(500)
    z = float(d.body('hei_object_cube_red').xpos[2])
    if abs(z-.82) > .005: raise AssertionError(f'Drop onto table failed: {z}')
    report['drop_onto_table_z_m'] = z
    place('cube_red',[.8,-.12,.821],(0,1,0))
    place('cube_green',[.8,0,.821])
    ids = {m.geom('hei_object_cube_red_geom').id,m.geom('hei_object_cube_green_geom').id}
    contacts = False
    for _ in range(300):
        mujoco.mj_step(m,d)
        contacts |= any({int(c.geom1),int(c.geom2)}==ids for c in d.contact)
    if not contacts: raise AssertionError('Object-object collision missing')
    report['object_object_contact'] = contacts
    friction=m.geom_friction.copy()
    distances={}
    try:
        for label,coefficient in [('low',.001),('high',.8)]:
            env.reset()
            m.geom_friction[:] = [coefficient,0,0]
            place('cube_red',[.9,-.42,.8201],(0,.35,0))
            step(150)
            distances[label]=float(d.body('hei_object_cube_red').xpos[1]+.42)
        if distances['low'] <= distances['high']+.04:
            raise AssertionError(f'Friction contrast failed: {distances}')
    finally:
        m.geom_friction[:]=friction
    report['slide_distance_m']=distances
    env.reset()
    initial_pos,initial_rot=d.cam_xpos.copy(),d.cam_xmat.copy()
    expected={'front':'lift_carriage_link','left_wrist':'b_left_link6','right_wrist':'a_right_link6'}
    for camera,parent in expected.items():
        if m.cam_bodyid[m.camera(camera).id] != m.body(parent).id:
            raise AssertionError(f'Incorrect mount: {camera}')
    for target_index,name in [(0,'right_wrist'),(7,'left_wrist')]:
        env.target[target_index] += .1 if target_index==0 else -.1
        env.step()
        cid=m.camera(name).id
        if np.linalg.norm(d.cam_xmat[cid]-initial_rot[cid]) < .02:
            raise AssertionError(f'Camera did not follow arm: {name}')
    if env.fixed_lift is None:
        env.target[17]=-.1
        for _ in range(10):env.step()
        if np.linalg.norm(d.cam_xpos[m.camera('front').id]-initial_pos[m.camera('front').id]) < .05:
            raise AssertionError('Front camera did not follow lift')
    if not np.allclose(d.cam_xpos[m.camera('overview').id],initial_pos[m.camera('overview').id]):
        raise AssertionError('External overview moved')
    report['camera_mounts_and_follow']=True
    if render:
        env.reset()
        obs=env.observe()
        if set(obs['images'])!=set(CAMERAS):raise AssertionError('Incorrect observation cameras')
        output=ROOT/'outputs/portable_check'
        output.mkdir(parents=True,exist_ok=True)
        for name,img in obs['images'].items():
            if img.shape!=(224,224,3) or np.std(img)<8:raise AssertionError(f'Bad render: {name}')
            write_png(output/(name+'.png'),img)
        write_png(output/'three_cameras.png',np.concatenate([obs['images'][k] for k in CAMERAS],axis=1))
        env.renderer.update_scene(d,camera='overview')
        write_png(output/'overview_not_observation.png',env.renderer.render().copy())
        report['render_checked']=True
        report['synchronized_three_camera_timestamp']=obs['timestamp']
    return report


def view(env, camera, seconds):
    import mujoco.viewer
    if sys.platform == 'darwin' and getattr(mujoco.viewer, '_MJPYTHON', None) is None:
        raise RuntimeError('On macOS run: mjpython -m hei_sim.portable --view')
    with mujoco.viewer.launch_passive(env.model,env.data) as viewer:
        viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid=env.model.camera(camera).id
        start=time.monotonic()
        while viewer.is_running() and (not seconds or time.monotonic()-start<seconds):
            frame_start=time.monotonic()
            with viewer.lock(): env.step()
            viewer.sync()
            time.sleep(max(0,1/env.fps-(time.monotonic()-frame_start)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-files',action='store_true')
    parser.add_argument('--check',action='store_true',help='Physics, camera mounts and offscreen renders')
    parser.add_argument('--physics-only',action='store_true',help='Skip render; does not certify graphics readiness')
    parser.add_argument('--grasp-check',action='store_true')
    parser.add_argument('--view',action='store_true')
    parser.add_argument('--camera',choices=(*CAMERAS,'overview'),default='overview')
    parser.add_argument('--seconds',type=float,default=0)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--fixed-lift',type=float,default=None)
    args=parser.parse_args()
    if args.verify_files: verify_manifest()
    if not any((args.check,args.physics_only,args.view,args.snapshot,args.grasp_check)):
        if not args.verify_files: parser.print_help()
        return
    env=Environment(fixed_lift=args.fixed_lift)
    try:
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True,exist_ok=True)
            args.snapshot.write_text(json.dumps(snapshot(env),indent=2),encoding='utf-8')
            print(f'Saved scene reference: {args.snapshot}',flush=True)
        if args.check or args.physics_only:
            report=check(env,render=not args.physics_only)
            output=ROOT/'outputs/portable_check'
            output.mkdir(parents=True,exist_ok=True)
            (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(report,indent=2),flush=True)
        if args.grasp_check:
            from .check_grasp import main as check_grasp
            env.close()
            check_grasp()
        if args.view:
            env.reset()
            view(env,args.camera,args.seconds)
    finally:
        env.close()


if __name__=='__main__':main()
