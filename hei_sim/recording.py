"""Transactional, synchronized three-camera episode storage."""
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

import cv2
import numpy as np

from .schema import CAMERAS, STATE_NAMES


def write_png(path, rgb):
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f'PNG encoding failed: {path}')
    encoded.tofile(str(path))


def read_png(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Invalid image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class EpisodeWriter:
    def __init__(self, root, metadata, task, test=False):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
        self.partial = root / (identifier + '.partial')
        self.final = root / identifier
        self.partial.mkdir()
        for name in CAMERAS:
            (self.partial / name).mkdir()
        self.metadata = {**metadata, 'format_version': 1, 'task': task, 'is_test': test,
                         'alignment': 'observation_t -> command_t -> physics_step', 'complete': False}
        (self.partial / 'metadata.json').write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')
        self.count = 0
        self.first_time = None
        self.closed = False

    def add(self, observation, action):
        if self.closed:
            raise RuntimeError('Episode already closed')
        if set(observation['images']) != set(CAMERAS):
            raise ValueError('Every frame requires exactly front, left_wrist and right_wrist')
        for name in CAMERAS:
            img = observation['images'][name]
            if img.dtype != np.uint8 or list(img.shape) != self.metadata['image_size']:
                raise ValueError(f'Invalid image {name}: {img.shape} {img.dtype}')
        state, action = np.asarray(observation['state']), np.asarray(action)
        if state.shape != (18,) or action.shape != (18,) or not np.isfinite([state, action]).all():
            raise ValueError('Invalid state/action')
        t = float(observation['timestamp'])
        if not np.isfinite(t):
            raise ValueError('Non-finite timestamp')
        if self.first_time is None:
            self.first_time = t
        elapsed = t - self.first_time
        if abs(elapsed - self.count / self.metadata['fps']) > 1e-5:
            raise ValueError('Non-uniform simulation timestamps')
        # A frame is committed only after all three PNG files exist. Exceptions
        # leave a .partial episode that converters will never train on.
        frame = f'{self.count:06d}'
        for name in CAMERAS:
            write_png(self.partial / name / (frame + '.png'), observation['images'][name])
        row = dict(frame_index=self.count, timestamp=elapsed, simulation_time=t,
                   state=state.tolist(), action=action.tolist())
        with (self.partial / 'frames.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        self.count += 1

    def save(self):
        if self.closed or not self.count:
            raise ValueError('Cannot save a closed or empty episode')
        self.metadata.update(complete=True, frames=self.count)
        (self.partial / 'metadata.json').write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')
        validate_episode(self.partial, allow_partial=True)
        self.partial.rename(self.final)
        self.closed = True
        return self.final

    def discard(self):
        # Keep interrupted/discarded material for inspection; excluded by converter.
        self.metadata.update(complete=False, discarded=True, frames=self.count)
        (self.partial / 'metadata.json').write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')
        self.closed = True


def validate_episode(path, allow_partial=False):
    path = Path(path)
    meta = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    if (path.suffix == '.partial' and not allow_partial) or not meta['complete']:
        raise ValueError('Incomplete episode')
    if meta['cameras'] != list(CAMERAS) or meta['state_names'] != STATE_NAMES or meta['action_names'] != STATE_NAMES:
        raise ValueError('Camera or state/action schema mismatch')
    expected_parents = {'front': 'lift_carriage_link', 'left_wrist': 'b_left_link6', 'right_wrist': 'a_right_link6'}
    for name, parent in expected_parents.items():
        if meta.get('camera_calibration', {}).get(name, {}).get('parent') != parent:
            raise ValueError(f'{name} must be mounted on {parent}; external overview is not training data')
    rows = [json.loads(line) for line in (path/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != meta['frames'] or not rows:
        raise ValueError('Frame count mismatch')
    expected = {f'{i:06d}.png' for i in range(len(rows))}
    for name in CAMERAS:
        if {p.name for p in (path/name).glob('*.png')} != expected:
            raise ValueError(f'Camera {name} missing or extra frames')
    for i, row in enumerate(rows):
        if not np.isfinite([row['timestamp'], row['simulation_time']]).all():
            raise ValueError('Non-finite timestamp')
        if (row['frame_index'] != i or abs(row['timestamp']-i/meta['fps']) > 1e-5
            or abs(row['simulation_time']-rows[0]['simulation_time']-row['timestamp']) > 1e-5):
            raise ValueError('Timestamp or index mismatch')
        if np.asarray(row['state']).shape != (18,) or np.asarray(row['action']).shape != (18,):
            raise ValueError('State/action size mismatch')
        if not np.isfinite([row['state'], row['action']]).all():
            raise ValueError('Non-finite state/action')
        for name in CAMERAS:
            if list(read_png(path/name/f'{i:06d}.png').shape) != meta['image_size']:
                raise ValueError('Image shape mismatch')
    return meta, rows
