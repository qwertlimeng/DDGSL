"""Small CPU-only checks; public experiment commands remain CUDA-only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from evaluate import evaluate_once
from models.ddgsl_config import CoreConfig
from models.ddgsl_model import DDGSL
from models.ddgsl_losses import (WindowMetadata, core_objective,
                                 direction_regularization, temporal_smoothness)
from src.config import load_config, read_json, require_cuda, write_json
from src.data import (WindowDataset, chronological_split, collate_windows,
                      fit_scaler, load_series, load_test_labels, transform)
from src.scoring import calibrate, point_metrics, score_test
from train import loss_step, train_once


class GenericPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_model_formula_gradients_and_variable_shapes(self):
        for window, nodes, depth in ((4, 3, 1), (8, 5, 2)):
            with self.subTest(window=window, nodes=nodes):
                torch.manual_seed(21)
                cfg = CoreConfig.confirmed(window, nodes, embedding_dim=4, topk=2,
                                           stib_depth=depth, mu1=.001, mu2=.003)
                model = DDGSL(cfg)
                values = np.random.RandomState(3).randn(window+5, nodes).astype('float32')
                segment = [{'start': 0, 'stop': len(values), 'sequence_id': 'a',
                            'segment_id': 0}]
                dataset = WindowDataset(values, segment, window=window, stride=2)
                batch = collate_windows([dataset[1], dataset[0]])
                result = model(batch['x'], return_details=True,
                    metadata=batch['metadata'], previous_x=batch['previous_x'],
                    previous_metadata=batch['previous_metadata'])
                self.assertEqual(tuple(result['prediction'].shape), (2, nodes))
                self.assertEqual(len(result['branches']), 2**(depth+1)-1)
                direction = result['branches'][0]
                sf, sb, sdir, beta = (direction[k] for k in ('S_f','S_b','S_dir','beta'))
                torch.testing.assert_close(sb, sf.transpose(-1, -2))
                torch.testing.assert_close(sdir-sdir.transpose(-1,-2),
                    (2*beta-1)*(sf-sf.transpose(-1,-2)), atol=1e-6, rtol=1e-6)
                lam = direction['lambda']
                torch.testing.assert_close(direction['B'], lam*direction['A_dir']+
                                           (1-lam)*direction['A_cos'])
                self.assertTrue((direction['A'] > 0).sum(-1).le(cfg.topk).all())
                terms = core_objective(result['prediction'], batch['target'],
                    result['L_smooth'], result['L_direction'], cfg.mu1, cfg.mu2)
                optimizer = torch.optim.Adam(model.parameters(), lr=.01)
                before = model.regression.weight.detach().clone()
                optimizer.zero_grad()
                terms['total'].backward()
                graphs = model.multiscale.shared_stib.gdn.graphs
                attention = model.multiscale.shared_stib.gdn.attention
                for parameter in (graphs.direction.embedding_f.weight,
                                  graphs.direction.embedding_b.weight,
                                  graphs.direction.forward_map.weight,
                                  graphs.direction.backward_map.weight,
                                  graphs.direction.beta_logit, graphs.direction.gamma,
                                  graphs.gate.weight, next(attention.parameters())):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertIsNotNone(next(model.multiscale.shared_stib.temporal.parameters()).grad)
                optimizer.step()
                self.assertFalse(torch.equal(before, model.regression.weight.detach()))

    def test_segments_previous_time_and_scaler_isolation(self):
        values = np.arange(60, dtype=np.float64).reshape(20, 3)
        segments = [{'start': 0, 'stop': 10, 'sequence_id': 'a', 'segment_id': 0},
                    {'start': 10, 'stop': 20, 'sequence_id': 'b', 'segment_id': 1}]
        train, val = chronological_split(segments, .6, window=2)
        self.assertEqual([(s['start'],s['stop']) for s in train], [(0,6),(10,16)])
        self.assertEqual([(s['start'],s['stop']) for s in val], [(6,10),(16,20)])
        scaler = fit_scaler(values, train, False)
        self.assertEqual(scaler['maximum'][0], values[15,0])
        transformed = transform(values, scaler)
        self.assertGreater(transformed[19,0], 1)
        identity = fit_scaler(values, train, True)
        np.testing.assert_array_equal(transform(values, identity), values.astype('float32'))
        dataset = WindowDataset(values.astype('float32'), train, window=2, stride=2)
        first, second, next_segment = dataset[0], dataset[1], dataset[2]
        self.assertEqual(second['time_index']-second['previous_metadata'].time_index.item(), 1)
        np.testing.assert_array_equal(second['previous_x'].numpy(), values[1:3].T)
        self.assertTrue(first['metadata'].is_first.item())
        self.assertTrue(next_segment['metadata'].is_first.item())
        shuffled = collate_windows([second, first])
        self.assertEqual(shuffled['previous_metadata'].time_index.tolist(), [3,2])
        self.assertEqual(next_segment['x'].shape, first['x'].shape)
        adjacent = WindowDataset(values.astype('float32'), train, window=2, stride=1)
        first_successor = adjacent[1]
        self.assertEqual(first_successor['previous_metadata'].window_index.item(), 0)
        self.assertTrue(first_successor['previous_metadata'].is_first.item())
        self.assertEqual(first_successor['time_index']-
                         first_successor['previous_metadata'].time_index.item(), 1)

    def test_csv_sequence_segments_and_label_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame({'sequence':['A']*10+['B']*10,
                                  'time':list(range(10))*2,
                                  'v1':range(20), 'v2':range(20,40)})
            frame.to_csv(root/'series.csv', index=False)
            series = load_series(root/'series.csv', sequence_id_column='sequence',
                                 timestamp_column='time', label_column='label')
            self.assertEqual(series['feature_columns'], ['v1','v2'])
            self.assertEqual([(s['start'],s['stop']) for s in series['segments']],
                             [(0,10),(10,20)])
            train, val = chronological_split(series['segments'], .6, window=2)
            self.assertEqual(WindowDataset(series['values'], val, window=2,
                                           stride=1).targets.tolist(), [8,9,18,19])
            frame['label'] = [0]*10+[1]*10
            frame.to_csv(root/'labelled.csv', index=False)
            labelled = load_series(root/'labelled.csv', sequence_id_column='sequence',
                                   timestamp_column='time', label_column='label')
            self.assertEqual(load_test_labels(labelled, root/'labelled.csv', None,
                                             'label').tolist(), [0]*10+[1]*10)

    def test_direction_regularizer_and_normal_only_calibration(self):
        a = torch.tensor([[[0.,2.],[0.,0.]]], requires_grad=True)
        direction, _, strength = direction_regularization(a, 1e-8)
        self.assertAlmostEqual(strength.item(), 2.)
        self.assertAlmostEqual(direction.item(), 1/3)
        direction.backward()
        self.assertIsNotNone(a.grad)
        val_p = np.array([[1.,2.],[2.,3.],[3.,4.]])
        val_t = np.zeros_like(val_p)
        calibration = calibrate(val_p, val_t, eps=1e-8, rho=.995)
        score, prediction = score_test(np.array([[20.,20.],[0.,0.]]),
                                       np.zeros((2,2)), calibration)
        self.assertEqual(calibration['theta'], float(np.quantile(calibration['validation_scores'], .995)))
        self.assertEqual(point_metrics(np.array([1,0]), score, prediction)['evaluated_points'], 2)

    def test_duplicate_headers_cannot_turn_labels_into_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for header in ('x,y,label,label', 'x,y,label, label', 'x,x,label'):
                path = root/'duplicate.csv'
                path.write_text(header+'\n'+','.join(['0']*len(header.split(',')))+'\n',
                                encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'unique'):
                    load_series(path, label_column='label', read_labels=True)
            pd.DataFrame({'x':[1,2], 'y':[3,4]}).to_csv(root/'test.csv',index=False)
            series = load_series(root/'test.csv',label_column='label')
            (root/'labels.csv').write_text('label,label\n0,1\n0,0\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unique'):
                load_test_labels(series,root/'test.csv',root/'labels.csv','label')

    def test_temporal_smoothness_uses_true_adjacent_graphs(self):
        current = torch.tensor([[[0.,1.],[0.,0.]], [[0.,2.],[0.,0.]]],
                               requires_grad=True)
        previous = torch.zeros_like(current)
        def metadata(index, time, first):
            return WindowMetadata(sequence_id=torch.tensor([0,0]),
                segment_id=torch.tensor([3,3]), window_index=torch.tensor(index),
                time_index=torch.tensor(time), stride=torch.tensor([1,1]),
                is_first=torch.tensor(first))
        now = metadata([0,1], [4,5], [True,False])
        prior = metadata([0,0], [4,4], [True,True])
        loss, per_window = temporal_smoothness(current, now, previous, prior)
        torch.testing.assert_close(per_window, torch.tensor([0., .5]))
        self.assertAlmostEqual(loss.item(), .25)
        loss.backward()
        self.assertNotEqual(current.grad[1].abs().sum().item(), 0.)

    def test_config_and_cuda_gate(self):
        path = Path(__file__).resolve().parents[1]/'configs'/'default.yaml'
        config = load_config(path)
        self.assertEqual(config['training']['seed'], 0)
        self.assertEqual(load_config(path, ['training.seed=7'])['training']['seed'], 7)
        with self.assertRaises(ValueError):
            load_config(path, ['training.unknown=1'])
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA GPU'):
                require_cuda()

    def test_tiny_end_to_end_with_and_without_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = np.random.RandomState(5)
            train = pd.DataFrame(generator.normal(size=(30,3)), columns=['v1','v2','v3'])
            test = pd.DataFrame(generator.normal(size=(18,3)), columns=['v1','v2','v3'])
            train.to_csv(root/'train.csv', index=False)
            test.to_csv(root/'test.csv', index=False)
            pd.DataFrame({'label':[0]*9+[1]*9}).to_csv(root/'labels.csv', index=False)
            config = {
                'dataset': {'train_path':'train.csv','test_path':'test.csv',
                    'test_label_path':'labels.csv','feature_columns':None,
                    'timestamp_column':None,'sequence_id_column':None,
                    'label_column':'label','normalized':False,'train_ratio':.7},
                'model': {'window_size':4,'top_k':2,'embedding_dim':4,'stib_depth':1},
                'training': {'batch_size':8,'train_stride':2,'learning_rate':.001,
                    'weight_decay':0.,'max_epochs':1,'patience':1,
                    'mu1':.001,'mu2':.003,'seed':0,'optimizer':'Adam'},
                'detection': {'rho':.995,'epsilon':1e-8,'evaluation_stride':1},
                'output_dir':'runs'}
            import yaml
            config_file = root/'config.yaml'
            config_file.write_text(yaml.safe_dump(config), encoding='utf-8')
            run_dir = train_once(config_file, device=torch.device('cpu'))
            effective = read_json(run_dir/'effective_config.json')
            manifest = read_json(run_dir/'split_manifest.json')
            self.assertEqual(effective['status'], 'complete')
            self.assertEqual(manifest['node_count'], 3)
            self.assertEqual(manifest['target_indices']['test']['count'], 14)
            output = evaluate_once(run_dir/'selected_model.pt', config_file,
                                   device=torch.device('cpu'))
            metrics = read_json(output/'metrics.json')
            self.assertTrue(metrics['labels_available'])
            self.assertIn('f1', metrics['metrics'])
            self.assertIn('auroc', metrics['metrics'])
            self.assertEqual(metrics['evaluated_points'], 14)
            calibration_path = run_dir/'calibration.json'
            saved = read_json(calibration_path)
            altered = dict(saved, theta=saved['theta']+1)
            write_json(calibration_path, altered)
            with self.assertRaisesRegex(ValueError, 'calibration changed'):
                evaluate_once(run_dir/'selected_model.pt', config_file,
                              device=torch.device('cpu'))
            write_json(calibration_path, saved)
            # A label-free run still outputs scores and decisions, never fake metrics.
            config['dataset']['test_label_path'] = None
            config_file.write_text(yaml.safe_dump(config), encoding='utf-8')
            no_label_run = train_once(config_file, device=torch.device('cpu'))
            no_label_output = evaluate_once(no_label_run/'selected_model.pt',
                                             config_file, device=torch.device('cpu'))
            no_label = read_json(no_label_output/'metrics.json')
            self.assertFalse(no_label['labels_available'])
            self.assertIsNone(no_label['metrics'])


if __name__ == '__main__':
    unittest.main()
