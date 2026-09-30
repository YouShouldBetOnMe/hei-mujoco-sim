"""Dynamic grip regression: hold, lift, release, and a low-friction control.

Gravity is disabled only while initializing a cube already between both jaws.
It is restored for every measured trial. There are no attachments or object
pose updates during hold/lift/release. This is not a full tabletop pick policy.
"""
import json

import mujoco
import numpy as np

from .environment import Environment, ROOT


def run_trial(env, side, local_x=.18, low_friction=False, legacy=False):
    model, data = env.model, env.data
    original_friction = model.geom_friction.copy()
    original_options = (model.opt.enableflags, model.opt.cone, model.opt.impratio, model.opt.noslip_iterations)
    prefix, start = ('a_right', 0) if side == 'right' else ('b_left', 7)
    finger_bodies = {model.body(prefix + '_end_link_' + s + '_finger').id for s in 'LR'}
    red = model.geom('hei_object_cube_red_geom').id
    joint = model.joint('hei_object_cube_red_free')
    q, v = joint.qposadr[0], joint.dofadr[0]
    if low_friction:
        # Both surfaces must change: equal-priority contact uses pairwise max.
        model.geom_friction[:] = [.001, 0, 0]
    if legacy:
        model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
        model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        model.opt.impratio = 1
        model.opt.noslip_iterations = 0
    try:
        env.reset()
        model.opt.gravity[:] = 0
        for _ in range(5):
            env.step()
        wrist = data.body(prefix + '_link6')
        pos = wrist.xpos + wrist.xmat.reshape(3,3) @ np.array([local_x,0,0])
        data.qpos[q:q+7] = [*pos, 1, 0, 0, 0]
        data.qvel[v:v+6] = 0
        env.target[start+6] = 0
        for _ in range(5):
            env.step()
        # End fixture initialization. All later object motion comes from mj_step.
        model.opt.gravity[:] = [0,0,-9.81]
        local_position = lambda: (data.qpos[q:q+3]-wrist.xpos) @ wrist.xmat.reshape(3,3)
        initial = local_position().copy()
        both_contacts, maximum_drift = 0, 0.
        for _ in range(100):
            env.step()
            contacting = set()
            for contact in data.contact:
                if red in (contact.geom1,contact.geom2):
                    other = contact.geom2 if contact.geom1 == red else contact.geom1
                    if model.geom_bodyid[other] in finger_bodies:
                        contacting.add(int(model.geom_bodyid[other]))
            both_contacts += contacting == finger_bodies
            maximum_drift = max(maximum_drift, float(np.linalg.norm(local_position()-initial)))
        result = dict(side=side, local_x_m=local_x, low_friction=low_friction, legacy=legacy,
            hold_seconds=10, maximum_relative_drift_m=maximum_drift,
            bilateral_contact_fraction=both_contacts/100)
        if not (legacy or low_friction):
            assert maximum_drift < .005, result
            assert both_contacts >= 98, result
            before_height = float(data.qpos[q+2])
            before_lift_local = local_position().copy()
            for _ in range(20):
                env.command(side, [0,0,.06], [0,0,0], 0)
                env.step()
            result['lift_height_m'] = float(data.qpos[q+2])-before_height
            result['lift_relative_drift_m'] = float(np.linalg.norm(local_position()-before_lift_local))
            assert result['lift_height_m'] > .06, result
            assert result['lift_relative_drift_m'] < .005, result
            before_release = float(data.qpos[q+2])
            env.target[start+6] = .1
            for _ in range(10):
                env.step()
            result['drop_after_open_m'] = before_release-float(data.qpos[q+2])
            assert result['drop_after_open_m'] > .15, result
        else:
            assert maximum_drift > .15, result
        return result
    finally:
        model.opt.gravity[:] = [0,0,-9.81]
        model.geom_friction[:] = original_friction
        model.opt.enableflags, model.opt.cone, model.opt.impratio, model.opt.noslip_iterations = original_options


def main():
    env = Environment()
    try:
        trials = []
        for side in ('right','left'):
            for x in (.16,.18,.20):
                result = run_trial(env,side,x)
                trials.append(result)
                print(json.dumps(result), flush=True)
        for kwargs in [dict(low_friction=True), dict(legacy=True)]:
            result = run_trial(env,'right',**kwargs)
            trials.append(result)
            print(json.dumps(result), flush=True)
        output = ROOT/'outputs/grasp_checks'
        output.mkdir(parents=True,exist_ok=True)
        report = dict(status='passed', scope='initialized bilateral grip, then dynamic hold/lift/release; not full tabletop pick',
                      physics=env.metadata()['physics'], trials=trials)
        (output/'grasp_regression.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:
        env.close()


if __name__ == '__main__':
    main()
