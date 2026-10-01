"""Download, verify and restore the Stack100 release using only Python's stdlib."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import urllib.request
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified(path, part):
    return path.is_file() and path.stat().st_size == part['size_bytes'] and sha256(path) == part['sha256']


def download(part, cache):
    target = cache / part['name']
    if verified(target, part):
        print(f"Verified cached {part['name']}", flush=True)
        return target
    if target.exists():
        raise ValueError(f'Cached archive differs: {target}; remove it before retrying.')
    partial = target.with_suffix('.zip.partial')
    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == part['size_bytes'] and verified(partial, part):
            partial.rename(target)
            return target
        if offset >= part['size_bytes']:
            partial.unlink()
            offset = 0
        headers = {'User-Agent': 'hei-mujoco-sim-dataset-downloader'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            request = urllib.request.Request(part['url'], headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                resume = response.status == 206
                if resume and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected download range')
                if not resume:
                    offset = 0
                last_progress = time.monotonic()
                with partial.open('ab' if resume else 'wb') as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        offset += len(block)
                        if offset > part['size_bytes']:
                            raise ValueError('Download exceeds expected size')
                        if time.monotonic() - last_progress >= 15:
                            print(f"{part['name']}: {100*offset/part['size_bytes']:.1f}%", flush=True)
                            last_progress = time.monotonic()
            if not verified(partial, part):
                if partial.stat().st_size == part['size_bytes']:
                    partial.unlink()
                raise ValueError('Incomplete download or SHA-256 mismatch')
            partial.rename(target)
            print(f"Downloaded and verified {part['name']}", flush=True)
            return target
        except (OSError, ValueError) as error:
            if attempt == 5:
                raise RuntimeError(f"Could not download {part['name']}: {error}") from None
            print(f"Retry {attempt}/5 for {part['name']}: {error}", flush=True)
            time.sleep(min(2**attempt, 20))


def extract(archive_path, stage, directory, part):
    with zipfile.ZipFile(archive_path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        if len(files) != part['file_count'] or sum(item.file_size for item in files) != part['uncompressed_bytes']:
            raise ValueError(f'Archive inventory differs: {archive_path.name}')
        for item in files:
            relative = PurePosixPath(item.filename)
            if (relative.is_absolute() or '..' in relative.parts or ':' in item.filename
                    or '\\' in item.filename or relative.parts[0] != directory
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError(f'Invalid archive member: {item.filename}')
            target = stage.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.extracting')
            with archive.open(item) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.replace(target)


def restore(manifest, name, output, cache, verify_only):
    variant = manifest['variants'][name]
    directory = variant['directory']
    if Path(directory).name != directory:
        raise ValueError('Invalid dataset directory')
    target = output / directory
    if not verify_only and target.exists():
        raise FileExistsError(f'Refusing to overwrite existing dataset: {target}')
    stage = output / ('.' + directory + '.unpacking')
    marker = stage / '_release.json'
    identity = dict(release_tag=manifest['release_tag'], variant=variant)
    if not verify_only:
        if stage.exists():
            if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8')) != identity:
                raise ValueError(f'Unrecognized staging directory: {stage}')
        else:
            stage.mkdir(parents=True)
            marker.write_text(json.dumps(identity), encoding='utf-8')
    for index, part in enumerate(variant['archives'], 1):
        if Path(part['name']).name != part['name']:
            raise ValueError('Invalid archive filename')
        print(f"{name}: archive {index}/{len(variant['archives'])}", flush=True)
        if verify_only:
            if not verified(cache / part['name'], part):
                raise ValueError(f"Archive missing or differs: {part['name']}")
        else:
            archive = download(part, cache)
            extract(archive, stage, directory, part)
    if verify_only:
        print(f'All {name} archive hashes match.', flush=True)
        return
    restored = stage / directory
    files = [path for path in restored.rglob('*') if path.is_file()]
    if len(files) != variant['file_count'] or sum(path.stat().st_size for path in files) != variant['uncompressed_bytes']:
        raise ValueError('Restored dataset inventory differs')
    restored.rename(target)
    marker.unlink()
    stage.rmdir()
    print(f'Dataset restored: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=['lerobot', 'raw', 'all'], default='lerobot')
    parser.add_argument('--output', type=Path, default=Path('datasets'))
    parser.add_argument('--archive-dir', type=Path, help='Archive cache; also accepts already downloaded ZIP files')
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'docs/datasets/dataset_release.json')
    parser.add_argument('--verify-only', action='store_true', help='Verify archives without downloading or extracting')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest['schema_version'] != 1 or manifest['archive_type'] != 'independent_zip_shards':
        raise ValueError('Unsupported release manifest')
    cache = args.archive_dir or args.output / '.downloads'
    if not args.verify_only:
        cache.mkdir(parents=True, exist_ok=True)
        args.output.mkdir(parents=True, exist_ok=True)
    names = ['lerobot', 'raw'] if args.format == 'all' else [args.format]
    for name in names:
        restore(manifest, name, args.output, cache, args.verify_only)


if __name__ == '__main__':
    main()
