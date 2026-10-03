"""Independent interpolation reference and optional deterministic CUDA check."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.ddgsl_config import CoreConfig
from models.ddgsl_losses import WindowMetadata, core_objective
from models.ddgsl_model import DDGSL
from models.recursive_stib import align_time


class DeterministicAlignmentTests(unittest.TestCase):
    def test_values_and_gradients_match_cpu_linear_interpolation(self):
        generator = torch.Generator().manual_seed(19)
        for dtype in (torch.float32, torch.float64):
            for source, target in ((1,8),(2,4),(4,2),(3,8),(8,3),(7,7),(16,32),(64,8)):
                with self.subTest(dtype=dtype, source=source, target=target):
                    x = torch.randn(2,3,source, dtype=dtype, generator=generator,
                                    requires_grad=True)
                    reference_x = x.detach().clone().requires_grad_(True)
                    upstream = torch.randn(2,3,target, dtype=dtype, generator=generator)
                    actual = align_time(x, target)
                    reference = F.interpolate(reference_x, size=target, mode='linear',
                                              align_corners=False)
                    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
                    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
                    actual.backward(upstream)
                    reference.backward(upstream)
                    torch.testing.assert_close(x.grad, reference_x.grad,
                                               atol=tolerance, rtol=tolerance)

    def test_model_predictions_and_gradients_preserve_reference_behavior(self):
        torch.manual_seed(21)
        cfg = CoreConfig.confirmed(8,3,embedding_dim=4,topk=2,stib_depth=2)
        model, reference = DDGSL(cfg), DDGSL(cfg)
        reference.load_state_dict(model.state_dict())
        x, target = torch.randn(2,3,8), torch.randn(2,3)
        actual = model(x)
        actual_loss = (actual-target).square().mean()
        actual_loss.backward()
        with patch('models.recursive_stib.align_time',
                   side_effect=lambda value, length: F.interpolate(
                       value, size=length, mode='linear', align_corners=False)):
            expected = reference(x)
            (expected-target).square().mean().backward()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        for (name, parameter), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=2e-5)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable in this interpreter')
    def test_cuda_strict_determinism_forward_backward_and_update(self):
        previous_mode = torch.are_deterministic_algorithms_enabled()
        previous_cudnn = torch.backends.cudnn.deterministic
        previous_benchmark = torch.backends.cudnn.benchmark
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            outcomes = []
            for _ in range(2):
                torch.manual_seed(21)
                cfg = CoreConfig.confirmed(4,3,embedding_dim=4,topk=2,stib_depth=1)
                model = DDGSL(cfg).cuda()
                values = torch.randn(7,3,device='cuda')
                x = torch.stack([values[1:5].T, values[2:6].T])
                prior = torch.stack([values[:4].T, values[1:5].T])
                def metadata(indices, times, first):
                    return WindowMetadata(
                        sequence_id=torch.tensor([0,0],device='cuda'),
                        segment_id=torch.tensor([0,0],device='cuda'),
                        window_index=torch.tensor(indices,device='cuda'),
                        time_index=torch.tensor(times,device='cuda'),
                        stride=torch.tensor([1,1],device='cuda'),
                        is_first=torch.tensor(first,device='cuda'))
                details = model(x, return_details=True,
                    metadata=metadata([1,2],[5,6],[False,False]), previous_x=prior,
                    previous_metadata=metadata([0,1],[4,5],[True,False]))
                terms = core_objective(details['prediction'], values[5:7],
                    details['L_smooth'], details['L_direction'], cfg.mu1, cfg.mu2)
                optimizer = torch.optim.Adam(model.parameters(),lr=.001)
                before = model.regression.weight.detach().clone()
                optimizer.zero_grad()
                terms['total'].backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                optimizer.step()
                torch.cuda.synchronize()
                self.assertFalse(torch.equal(before,model.regression.weight.detach()))
                outcomes.append((details['prediction'].detach().cpu(),
                    {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}))
            self.assertTrue(torch.equal(outcomes[0][0],outcomes[1][0]))
            for name in outcomes[0][1]:
                self.assertTrue(torch.equal(outcomes[0][1][name],outcomes[1][1][name]),name)
        finally:
            torch.use_deterministic_algorithms(previous_mode)
            torch.backends.cudnn.deterministic = previous_cudnn
            torch.backends.cudnn.benchmark = previous_benchmark


if __name__ == '__main__':
    unittest.main()
