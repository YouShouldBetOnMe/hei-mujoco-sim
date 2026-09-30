"""Rebuild packaged asset hashes after an intentional scene resource change."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
package = root / 'hei_sim'
files = [p for p in (package/'assets').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [package/'gamepad_config.json', package/'scene_reference.json']
manifest = dict(schema_version=1, source_repository='https://github.com/lipengdong/hei-rebot-lift',
    source_commit='40674132257099656a26ee7fd11cb67f80f25658', mujoco_version='3.5.0',
    sha256={p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
(package/'ASSET_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(f'Hashed {len(files)} package resources')
