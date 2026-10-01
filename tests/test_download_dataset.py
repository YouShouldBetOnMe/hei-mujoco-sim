import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request
import zipfile


spec = importlib.util.spec_from_file_location('dataset_downloader',
    Path(__file__).resolve().parents[1] / 'scripts/download_dataset.py')
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='hei_download_test_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = bytes(range(256)) * 100
        self.ranges = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requested = self.headers.get('Range')
                owner.ranges.append(requested)
                start = int(requested.removeprefix('bytes=').removesuffix('-')) if requested else 0
                resume = bool(requested) and self.path == '/resume'
                self.send_response(206 if resume else 200)
                body = owner.payload[start:] if resume else owner.payload
                if resume:
                    self.send_header('Content-Range', f'bytes {start}-{len(owner.payload)-1}/{len(owner.payload)}')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.request_patch = patch.object(downloader.urllib.request, 'urlopen', opener.open)
        self.request_patch.start()
        self.addCleanup(self.request_patch.stop)

    def part(self, endpoint='/resume'):
        return dict(name='test.zip', size_bytes=len(self.payload),
                    sha256=hashlib.sha256(self.payload).hexdigest(),
                    url=f'http://127.0.0.1:{self.server.server_port}{endpoint}')

    def test_resumes_an_interrupted_download(self):
        (self.root / 'test.zip.partial').write_bytes(self.payload[:777])
        restored = downloader.download(self.part(), self.root)
        self.assertEqual(restored.read_bytes(), self.payload)
        self.assertEqual(self.ranges, ['bytes=777-'])

    def test_restarts_when_server_ignores_range(self):
        (self.root / 'test.zip.partial').write_bytes(self.payload[:777])
        restored = downloader.download(self.part('/ignore'), self.root)
        self.assertEqual(restored.read_bytes(), self.payload)

    def test_bad_hash_never_becomes_a_completed_archive(self):
        part = {**self.part(), 'sha256': '0' * 64}
        with patch.object(downloader.time, 'sleep'), self.assertRaises(RuntimeError):
            downloader.download(part, self.root)
        self.assertFalse((self.root / 'test.zip').exists())
        self.assertFalse((self.root / 'test.zip.partial').exists())

    def test_extraction_rejects_paths_outside_dataset(self):
        archive = self.root / 'outside.zip'
        with zipfile.ZipFile(archive, 'w') as output:
            output.writestr('dataset/../../outside.txt', b'bad')
        with self.assertRaises(ValueError):
            downloader.extract(archive, self.root / 'stage', 'dataset',
                               dict(file_count=1, uncompressed_bytes=3))
        self.assertFalse((self.root / 'outside.txt').exists())

    def test_existing_dataset_is_preserved(self):
        existing = self.root / 'dataset'
        existing.mkdir()
        (existing / 'keep.txt').write_text('keep', encoding='utf-8')
        manifest = dict(release_tag='test', variants={'raw': dict(directory='dataset', archives=[])})
        with self.assertRaises(FileExistsError):
            downloader.restore(manifest, 'raw', self.root, self.root, False)
        self.assertEqual((existing / 'keep.txt').read_text(), 'keep')

    def test_completed_format_is_skipped_when_resuming_all(self):
        cache = self.root / 'cache'
        cache.mkdir()
        archive = cache / 'test.zip'
        with zipfile.ZipFile(archive, 'w') as output:
            output.writestr('dataset/frame.txt', b'frame')
        part = dict(name=archive.name, size_bytes=archive.stat().st_size,
                    sha256=downloader.sha256(archive), file_count=1, uncompressed_bytes=5)
        manifest = dict(release_tag='test', variants={'raw': dict(directory='dataset',
                        file_count=1, uncompressed_bytes=5, archives=[part])})
        output = self.root / 'output'
        downloader.restore(manifest, 'raw', output, cache, False)
        downloader.restore(manifest, 'raw', output, cache, False)
        self.assertEqual((output / 'dataset/frame.txt').read_bytes(), b'frame')


if __name__ == '__main__':
    unittest.main()
