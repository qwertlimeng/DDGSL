"""Synthetic checks for CSV preparation; no real dataset or GPU training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from preprocess import prepare_data
from src.config import load_config
from src.data import file_sha256, load_series
from train import prepare


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def inputs(self, train, test=None):
        train.to_csv(self.root/'normal.csv', index=False)
        (train if test is None else test).to_csv(self.root/'attack.csv', index=False)

    def run_prepare(self, name='out', features=('x',), **kwargs):
        return prepare_data(self.root/'normal.csv', self.root/'attack.csv',
                            self.root/name, feature_columns=features, **kwargs)

    def test_51_features_preserved_and_training_scaler_isolated(self):
        names = ['v'+str(i) for i in range(50)] + ['P603']
        values = np.arange(40*51, dtype=float).reshape(40, 51)
        train = pd.DataFrame(values, columns=names)
        test = pd.DataFrame(values[:20]+100000, columns=names)
        train['attack'], test['attack'] = 0, [0]*10+[1]*10
        self.inputs(train, test[list(reversed(names))+['attack']])
        hashes = [file_sha256(self.root/name) for name in ('normal.csv', 'attack.csv')]
        manifest = self.run_prepare(features=names, label_column='attack')
        output = pd.read_csv(self.root/'out'/'train.csv')
        np.testing.assert_array_equal(output[names].values, values)
        np.testing.assert_array_equal(pd.read_csv(self.root/'out'/'test.csv')[names].values,
                                      values[:20]+100000)
        self.assertEqual(list(output.columns[:51]), names)
        config_file = Path(__file__).resolve().parents[1]/'configs'/'default.yaml'
        config = load_config(config_file)
        config['dataset'] = manifest['dataset_config']
        config['model'].update(window_size=4, top_k=2, stib_depth=1)
        core, normal, test_series, scaler, windows, validation, split = prepare(config, config_file)
        self.assertEqual(core.node_num, 51)
        self.assertEqual(split['train_segments'][0]['stop'], 28)
        self.assertEqual(split['validation_segments'][0]['start'], 28)
        np.testing.assert_array_equal(scaler['maximum'], values[27])
        self.assertEqual(validation.targets[0], 32)
        self.assertEqual(windows[1]['previous_metadata'].time_index.item(), 8)
        for name, checksum in zip(('normal.csv', 'attack.csv'), hashes):
            self.assertEqual(file_sha256(self.root/name), checksum)
        for name, checksum in manifest['output_sha256'].items():
            self.assertEqual(file_sha256(self.root/'out'/name), checksum)
        mapping = pd.read_csv(self.root/'out'/'train_row_mapping.csv')
        self.assertEqual(mapping.iloc[32]['source_row_start'], 32)
        self.assertEqual(mapping.iloc[32]['source_row_stop'], 33)

    def test_medians_labels_timestamps_and_contiguous_segments(self):
        frame = pd.DataFrame({'x': [0,2,4,100,102,104,200,202,204],
                              'unit': ['a']*3+['b']*3+['a']*3,
                              'time': list(range(3))*3, 'label': [0]*9})
        test = frame.copy()
        test['label'] = [0,1,0,0,0,1,0,1,0]
        self.inputs(frame, test)
        manifest = self.run_prepare(downsample=2, tail='keep',
                                    sequence_id_column='unit', timestamp_column='time')
        output = pd.read_csv(self.root/'out'/'test.csv')
        self.assertEqual(output.x.tolist(), [1,4,101,104,201,204])
        self.assertEqual(output.label.tolist(), [1,0,0,1,1,0])
        self.assertEqual(output.timestamp.tolist(), [1,2,1,2,1,2])
        self.assertEqual(output.segment_id.tolist(), [0,0,1,1,2,2])
        series = load_series(self.root/'out'/'test.csv', feature_columns=['x'],
            sequence_id_column='segment_id', timestamp_column='timestamp', label_column='label')
        self.assertEqual(len(series['segments']), 3)
        self.assertFalse(manifest['historical_paper_provenance_verified'])
        with self.assertRaisesRegex(ValueError, 'single'):
            self.run_prepare(name='trim', downsample=2, tail='keep',
                sequence_id_column='unit', drop_train_prefix_blocks=1)

    def test_explicit_tail_and_train_only_prefix(self):
        self.inputs(pd.DataFrame({'x': range(7), 'label': [0]*7}))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.run_prepare(downsample=2)
        self.assertFalse((self.root/'out').exists())
        manifest = self.run_prepare(downsample=2, tail='drop', drop_train_prefix_blocks=1)
        self.assertEqual(pd.read_csv(self.root/'out'/'train.csv').x.tolist(), [2.5,4.5])
        self.assertEqual(pd.read_csv(self.root/'out'/'test.csv').x.tolist(), [.5,2.5,4.5])
        summary = manifest['source_files']['train']['segments'][0]
        self.assertEqual(summary['dropped_prefix_rows'], 2)
        self.assertEqual(summary['dropped_tail_rows'], 1)
        mapping = pd.read_csv(self.root/'out'/'train_row_mapping.csv')
        self.assertEqual(mapping.source_row_start.tolist(), [2,4])
        self.assertEqual(mapping.source_row_stop.tolist(), [4,6])
        with self.assertRaises(FileExistsError):
            self.run_prepare()

    def test_forward_fill_is_causal_and_never_crosses_segments(self):
        train = pd.DataFrame({'x':[1,np.nan,999,7,np.nan,9],
                              'unit':['a']*3+['b']*3, 'label':[0]*6})
        test = train.copy()
        test['x'] = [10,np.nan,100000,20,np.nan,200000]
        self.inputs(train, test)
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.run_prepare(sequence_id_column='unit')
        self.run_prepare(missing='forward-fill', sequence_id_column='unit')
        self.assertEqual(pd.read_csv(self.root/'out'/'train.csv').x.tolist(), [1,1,999,7,7,9])
        self.assertEqual(pd.read_csv(self.root/'out'/'test.csv').x.tolist(), [10,10,100000,20,20,200000])
        test.loc[3, 'x'] = np.nan
        self.inputs(train, test)
        with self.assertRaisesRegex(ValueError, 'segment-leading'):
            self.run_prepare(name='leading', missing='forward-fill', sequence_id_column='unit')

    def test_bad_labels_features_headers_and_normal_training_rejected(self):
        for bad in ([0,2], [0,np.nan], ['Normal','Attack']):
            self.inputs(pd.DataFrame({'x':[1,2], 'label':[0,0]}),
                        pd.DataFrame({'x':[3,4], 'label':bad}))
            with self.assertRaises(ValueError):
                self.run_prepare()
        self.inputs(pd.DataFrame({'x':[1,2], 'label':[0,1]}))
        with self.assertRaisesRegex(ValueError, 'normal'):
            self.run_prepare()
        self.inputs(pd.DataFrame({'x':[1,np.inf], 'label':[0,0]}))
        with self.assertRaisesRegex(ValueError, 'infinite'):
            self.run_prepare()
        self.inputs(pd.DataFrame({'x':[1e308,1e308], 'label':[0,0]}))
        with self.assertRaisesRegex(ValueError, 'nonfinite block median'):
            self.run_prepare(downsample=2)
        self.inputs(pd.DataFrame({'x':[1,2], 'label':[0,0]}))
        with self.assertRaisesRegex(ValueError, 'missing requested'):
            self.run_prepare(features=['P603'])
        (self.root/'normal.csv').write_text('x, x,label\n1,2,0\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.run_prepare()

    def test_cli_feature_list_and_already_normalized_declaration(self):
        self.inputs(pd.DataFrame({'x':[.1,.2,.3,.4], 'attack':[0]*4}))
        feature_list = self.root/'features.txt'
        feature_list.write_text('x\n', encoding='utf-8')
        script = Path(__file__).resolve().parents[1]/'preprocess.py'
        result = subprocess.run([sys.executable, '-B', str(script), '--train', str(self.root/'normal.csv'),
            '--test', str(self.root/'attack.csv'), '--output-dir', str(self.root/'out'),
            '--feature-list', str(feature_list), '--label-column', 'attack',
            '--already-normalized'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('No training performed', result.stdout)
        manifest = json.loads((self.root/'out'/'preprocessing_manifest.json').read_text(encoding='utf-8'))
        self.assertTrue(manifest['dataset_config']['normalized'])
        self.assertIsNone(manifest['dataset_config']['test_label_path'])
        np.testing.assert_allclose(pd.read_csv(self.root/'out'/'train.csv').x, [.1,.2,.3,.4])


if __name__ == '__main__':
    unittest.main()
