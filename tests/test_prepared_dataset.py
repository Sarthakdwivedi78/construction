import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from prepare_dataset import build
from train_vision_lora import load_splits


class PreparedDatasetTests(unittest.TestCase):
    def test_prepare_train_contract_and_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = {}
            for i in range(1000):
                site = f'fixture-site-{i}'
                bucket = int(hashlib.sha256(site.encode()).hexdigest()[:8], 16) % 100
                split = 'train' if bucket < 70 else 'validation' if bucket < 85 else 'test'
                if split in rows:
                    continue
                image = root / f'{i}.png'
                image.write_bytes(f'fixture bytes {i}'.encode())
                rows[split] = dict(case_id=str(i), image=image.name, site_id=site,
                    source_url='https://example.org/fixture', usage_permission='synthetic fixture',
                    reviewer='test fixture', reviewed=True, question='Fixture question',
                    retrieved_regulations=[], answer=dict(hazard='Unknown fixture hazard',
                    violation_status='INSUFFICIENT_EVIDENCE', reasoning='Insufficient fixture evidence',
                    corrective_action='Request evidence'))
                if len(rows) == 3:
                    break
            manifest = root / 'reviewed.jsonl'
            def write_manifest():
                manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows.values()), encoding='utf-8')
            write_manifest()
            destination = root / 'splits'
            self.assertEqual(build(manifest, root, destination), dict(train=1, validation=1, test=1))
            prepared = load_splits(destination, root)
            self.assertEqual(set(prepared), {'train', 'validation', 'test'})
            for records in prepared.values():
                self.assertEqual(len(records[0]['sha256']), 64)
            originals = {p: p.read_bytes() for p in destination.iterdir()}
            with self.assertRaisesRegex(ValueError, 'already exist'):
                build(manifest, root, destination)
            self.assertTrue(all(p.read_bytes() == data for p, data in originals.items()))
            del rows['train']['case_id']
            write_manifest()
            with self.assertRaisesRegex(ValueError, 'Case IDs'):
                build(manifest, root, root / 'missing-id')
            self.assertFalse((root / 'missing-id').exists())
            image = Path(prepared['train'][0]['image_path'])
            image.write_bytes(b'changed after preparation')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                load_splits(destination, root)
