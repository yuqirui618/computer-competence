"""ASL on Study mean probabilities, never on averaged image logits.

Sources:
https://raw.githubusercontent.com/Alibaba-MIIL/ASL/main/src/loss_functions/losses.py
https://arxiv.org/html/2009.14119 (sections 2.2--2.5)

Default (gamma_neg=2, gamma_pos=0, clip=.05) is a mild experimental
configuration, NOT the author's default (4, 1, .05).  Focal weights are
stopped by default, matching the ordinary AsymmetricLoss class's
``disable_torch_grad_focal_loss=True``; the paper's derivative also
contains the weight derivative, available here by setting False.

Inputs are [B,C] log_p=log(mean_v(sigmoid(z_v))) and
log_q=log(mean_v(sigmoid(-z_v))), computed over ALL valid Study images.
Use logsumexp(-softplus(-/+z_v)) - log(n_v) for stable pooling.  We do
not apply sigmoid, average logits, clip probabilities to eps, or count
class positives.  All math is FP32 with autocast disabled.  Finite log
probabilities down to -1e30 are preserved; -inf endpoints / FP32 overflow
are capped at -1e30, and positive roundoff is capped at zero.

Reduction 'none' returns each Study's class MEAN [B], so the runner must
backpropagate per_study.sum()/actual_accumulation_window_studies.  For a
17-Study window split 16+1, do NOT divide each microbatch mean by 2.
'mean' and 'sum' then reduce this [B] vector.

Adapted formulas; MIT License follows.
Copyright (c) 2020 Alibaba-MIIL
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import math
import torch
from torch import nn

SOURCE_CODE_URL = 'https://raw.githubusercontent.com/Alibaba-MIIL/ASL/main/src/loss_functions/losses.py'
PAPER_URL = 'https://arxiv.org/html/2009.14119'
LOG_PROB_MIN = -1e30


class ASLStudyLoss(nn.Module):
    """Binary multi-hot [B,C] log probabilities -> class-mean Study losses.

    targets must contain 0/1 and share the inputs' device. 'none' -> [B];
    'mean' -> mean over Studies; 'sum' -> sum of class-mean Study losses.
    Focal stopping is local detach and never changes global grad state.
    """

    def __init__(self, gamma_neg=2.0, gamma_pos=0.0, clip=0.05,
                 disable_focal_grad=True, reduction='none'):
        super().__init__()
        if not math.isfinite(gamma_neg) or not math.isfinite(gamma_pos) or min(gamma_neg, gamma_pos) < 0:
            raise ValueError('gammas must be finite and nonnegative')
        if not math.isfinite(clip) or not 0 <= clip < 1:
            raise ValueError('clip must be finite in [0,1)')
        if not isinstance(disable_focal_grad, bool):
            raise TypeError('disable_focal_grad must be an explicit bool')
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('reduction must be none, mean, or sum')
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.disable_focal_grad = disable_focal_grad
        self.reduction = reduction

    def forward(self, log_p, log_q, targets):
        if log_p.ndim != 2 or log_q.shape != log_p.shape or targets.shape != log_p.shape:
            raise ValueError('log_p, log_q, targets must share [B,C] shape')
        if not log_p.shape[0] or not log_p.shape[1]:
            raise ValueError('B and C must be positive')
        if log_p.device != log_q.device or targets.device != log_p.device:
            raise ValueError('all inputs must share a device')
        with torch.autocast(device_type=log_p.device.type, enabled=False):
            lp = log_p.float().clamp(min=LOG_PROB_MIN, max=0.0)
            lq = log_q.float().clamp(min=LOG_PROB_MIN, max=0.0)
            y = targets.float()
            if self.clip:
                lq = torch.logaddexp(lq, lq.new_tensor(math.log(self.clip))).clamp(max=0.0)
            positive = y.bool()  # Input contract is binary multi-hot, not soft labels.
            log_pt = torch.where(positive, lp, lq)
            loss = -log_pt
            if self.gamma_neg > 0 or self.gamma_pos > 0:
                gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
                # expm1 protects 1-pt when pt is very close to one.
                base = -torch.expm1(log_pt)
                # Branch before pow: avoid 0**0 and inf*0 gradients for
                # fractional gamma at clipped/easy zero-weight endpoints.
                has_base = base > 0
                safe_base = torch.where(has_base, base, torch.ones_like(base))
                weight = torch.where(has_base | (gamma == 0), safe_base.pow(gamma),
                                     torch.zeros_like(base))
                if self.disable_focal_grad:
                    weight = weight.detach()
                loss = loss * weight
            per_study = loss.mean(dim=1)
            if self.reduction == 'mean':
                return per_study.mean()
            if self.reduction == 'sum':
                return per_study.sum()
            return per_study

    def extra_repr(self):
        return (f'gamma_neg={self.gamma_neg}, gamma_pos={self.gamma_pos}, '
                f'clip={self.clip}, disable_focal_grad={self.disable_focal_grad}, '
                f'reduction={self.reduction!r}')


def _cpu_checks():
    """Deterministic CPU-only acceptance checks; no model, GPU, or training."""
    import datetime
    import hashlib
    import json
    import sys
    import traceback
    import urllib.request
    from pathlib import Path
    from torch.nn import functional as F

    torch.set_num_threads(2)
    torch.manual_seed(20261106)
    checks = {}
    root = Path(__file__).resolve().parent
    receipt = {
        'status': 'running', 'device': 'cpu', 'torch_version': str(torch.__version__),
        'python': sys.executable, 'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_urls': {'code': SOURCE_CODE_URL, 'paper': PAPER_URL,
                        'license': 'https://raw.githubusercontent.com/Alibaba-MIIL/ASL/main/LICENSE'},
        'configuration': {'gamma_neg': 2.0, 'gamma_pos': 0.0, 'clip': 0.05,
                          'disable_focal_grad': True, 'reduction': 'none'},
        'not_author_default_parameters': True,
        'author_ordinary_class_defaults': {'gamma_neg': 4, 'gamma_pos': 1, 'clip': 0.05,
                                           'eps': 1e-8, 'disable_torch_grad_focal_loss': True},
        'reduction': 'mean over C per Study; reduction=none returns [B]',
        'endpoint_policy': {'min_log_probability': LOG_PROB_MIN, 'max_log_probability': 0.0,
                            'author_eps_clamp_removed': True},
        'input_contract': 'binary multi-hot targets; complementary stable logs of mean image probabilities; all valid images included',
        'accumulation_rule': 'per_study.sum()/actual_window_studies; final 17 = 16+1',
        'gpu_used': False, 'model_constructed': False, 'training_run': False,
        'checks': checks,
    }

    def close(a, b, atol=2e-6, rtol=2e-5):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        return float((a.detach().float() - b.detach().float()).abs().max())

    def pooled(z, lengths):
        lp, lq = [], []
        for part in z.split(lengths):
            lp.append(torch.logsumexp(-F.softplus(-part), dim=0) - math.log(len(part)))
            lq.append(torch.logsumexp(-F.softplus(part), dim=0) - math.log(len(part)))
        return torch.stack(lp), torch.stack(lq)

    def loss_from_logits(z, y, criterion):
        return criterion(-F.softplus(-z), -F.softplus(z), y)

    try:
        p = torch.tensor([[0.2, 0.4, 0.8, 0.01]], dtype=torch.float32)
        y = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        z = torch.logit(p).requires_grad_()
        actual = loss_from_logits(z, y, ASLStudyLoss())
        p_values = p[0].tolist()
        terms = [-math.log(p_values[0]), -(p_values[1]-.05)**2 * math.log(1-p_values[1]+.05),
                 -math.log(p_values[2]), 0.0]
        expected = torch.tensor([sum(terms)/4])
        actual_grad, = torch.autograd.grad(actual.sum(), z)
        expected_grad = torch.tensor([[(p_values[0]-1)/4,
             (p_values[1]-.05)**2*p_values[1]*(1-p_values[1])/(1-p_values[1]+.05)/4,
             (p_values[2]-1)/4, 0.0]])
        checks['hand_value_and_detached_gradient'] = {'passed': True,
            'value_max_abs_error': close(actual, expected),
            'gradient_max_abs_error': close(actual_grad, expected_grad), 'loss': actual.tolist()}

        z = torch.randn(17, 10).requires_grad_()
        y = torch.randint(0, 2, z.shape).float()
        degenerate = loss_from_logits(z, y, ASLStudyLoss(0, 0, 0))
        bce = F.binary_cross_entropy_with_logits(z, y, reduction='none').mean(dim=1)
        g1, = torch.autograd.grad(degenerate.sum(), z, retain_graph=True)
        g2, = torch.autograd.grad(bce.sum(), z)
        checks['zero_gamma_zero_clip_equals_bce'] = {'passed': True,
            'value_max_abs_error': close(degenerate, bce), 'gradient_max_abs_error': close(g1, g2)}

        lengths = [1, 2, 4, 3, 1, 5, 2, 1, 3, 2, 4, 1, 2, 3, 1, 2, 7]
        images = (torch.randn(sum(lengths), 10) * 1.7).requires_grad_()
        lp, lq = pooled(images, lengths)
        pp = torch.stack([part.sigmoid().mean(0) for part in images.split(lengths)])
        bce_prob = F.binary_cross_entropy(pp, y, reduction='none').mean(dim=1)
        study_bce = ASLStudyLoss(0, 0, 0)(lp, lq, y)
        gg1, = torch.autograd.grad(study_bce.sum(), images, retain_graph=True)
        gg2, = torch.autograd.grad(bce_prob.sum(), images)
        checks['complete_image_probability_mean_bce'] = {'passed': True,
            'study_lengths': lengths, 'pool_probability_max_abs_error': close(lp.exp(), pp),
            'pool_complement_max_abs_error': close(lp.exp()+lq.exp(), torch.ones_like(pp)),
            'value_max_abs_error': close(study_bce, bce_prob), 'gradient_max_abs_error': close(gg1, gg2)}

        source = urllib.request.urlopen(SOURCE_CODE_URL, timeout=30).read()
        receipt['upstream_code_sha256'] = hashlib.sha256(source).hexdigest()
        namespace = {}
        # Exactly the ordinary author's class, including its original gradient flag code.
        exec(compile(source.decode('utf-8').split('class AsymmetricLossOptimized')[0],
                     SOURCE_CODE_URL, 'exec'), namespace)
        author_class = namespace['AsymmetricLoss']
        comparisons = []
        for gamma_neg, gamma_pos in ((2.0, 0.0), (4.0, 1.0)):
            for stopped in (True, False):
                z = torch.linspace(-6, 6, 170).reshape(17, 10).requires_grad_()
                ours = loss_from_logits(z, y, ASLStudyLoss(gamma_neg, gamma_pos, .05, stopped)).mean()
                author = author_class(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=.05,
                    disable_torch_grad_focal_loss=stopped)(z, y) / z.numel()
                go, = torch.autograd.grad(ours, z, retain_graph=True)
                ga, = torch.autograd.grad(author, z)
                comparisons.append({'gamma_neg': gamma_neg, 'gamma_pos': gamma_pos,
                    'disable_focal_grad': stopped, 'value_max_abs_error': close(ours, author),
                    'gradient_max_abs_error': close(go, ga)})
        checks['author_moderate_logits_forward_and_gradient'] = {'passed': True, 'cases': comparisons,
            'scope': 'fp32 logits in [-6,6]; author sum divided by numel; matched focal-grad setting'}

        cases = {}
        for label, yy in [('all_positive', torch.ones(3,10)), ('all_negative', torch.zeros(3,10)),
                          ('rare_class_no_positive', torch.randint(0,2,(3,10)).float())]:
            if label == 'rare_class_no_positive':
                yy[:, -1] = 0
            z = torch.randn(3, 10).requires_grad_()
            value = loss_from_logits(z, yy, ASLStudyLoss())
            grad, = torch.autograd.grad(value.sum(), z)
            assert value.shape == (3,) and torch.isfinite(value).all() and torch.isfinite(grad).all()
            cases[label] = {'losses': value.tolist(), 'max_abs_gradient': float(grad.abs().max())}
            if label == 'all_positive':
                close(value, F.softplus(-z).mean(dim=1))
        checks['label_composition_safety'] = {'passed': True, 'cases': cases,
            'uses_no_inverse_class_counts': True}

        z = torch.tensor([[-100., 100., -100., 100.]], requires_grad=True)
        yy = torch.tensor([[0.,1.,1.,0.]])
        value = loss_from_logits(z, yy, ASLStudyLoss(.5,.5,.05,False))
        grad, = torch.autograd.grad(value.sum(), z)
        assert torch.isfinite(value).all() and torch.isfinite(grad).all()
        checks['fractional_gamma_zero_weight_gradient_safety'] = {'passed': True,
            'gamma_neg': .5, 'gamma_pos': .5, 'disable_focal_grad': False,
            'loss': value.tolist(), 'max_abs_gradient': float(grad.abs().max())}

        z = torch.tensor([[-1e4,-100.,-20.,0.,20.,100.,1e4],
                          [-1e4,-100.,-20.,0.,20.,100.,1e4]], requires_grad=True)
        yy = torch.stack([torch.zeros(7), torch.ones(7)])
        extreme_cases = []
        for stopped in (True, False):
            value = loss_from_logits(z, yy, ASLStudyLoss(disable_focal_grad=stopped))
            grad, = torch.autograd.grad(value.sum(), z, retain_graph=True)
            assert torch.isfinite(value).all() and torch.isfinite(grad).all()
            extreme_cases.append({'disable_focal_grad': stopped, 'losses': value.tolist(),
                                  'max_abs_gradient': float(grad.abs().max())})
        extreme_bce = loss_from_logits(z, yy, ASLStudyLoss(0,0,0))
        stable_bce = F.binary_cross_entropy_with_logits(z, yy, reduction='none').mean(1)
        g1, = torch.autograd.grad(extreme_bce.sum(), z, retain_graph=True)
        g2, = torch.autograd.grad(stable_bce.sum(), z)
        checks['extreme_finite_log_probabilities'] = {'passed': True, 'logit_range': [-1e4,1e4],
            'cases': extreme_cases, 'bce_value_max_abs_error': close(extreme_bce, stable_bce),
            'bce_gradient_max_abs_error': close(g1,g2)}

        lp = torch.tensor([[-float('inf'), 0.]], requires_grad=True)
        lq = torch.tensor([[0., -float('inf')]], requires_grad=True)
        value = ASLStudyLoss(0,0,0)(lp,lq,torch.tensor([[1.,0.]]))
        gradp, gradq = torch.autograd.grad(value.sum(), (lp,lq))
        assert torch.isfinite(value).all() and torch.isfinite(gradp).all() and torch.isfinite(gradq).all()
        checks['exact_zero_one_log_endpoint_guard'] = {'passed': True, 'loss': value.tolist(),
                                                     'finite_cap': -LOG_PROB_MIN}

        image_features = torch.randn(sum(lengths), 6)
        yy = torch.randint(0,2,(17,10)).float()
        initial = torch.randn(6,10) * .15
        full_w = initial.clone().requires_grad_()
        lp,lq = pooled(image_features @ full_w, lengths)
        full_loss = ASLStudyLoss()(lp,lq,yy).mean()
        full_grad, = torch.autograd.grad(full_loss, full_w)
        split_w = initial.clone().requires_grad_()
        offset = 0
        for begin, end in ((0,16),(16,17)):
            sizes = lengths[begin:end]
            count = sum(sizes)
            lp,lq = pooled(image_features[offset:offset+count] @ split_w, sizes)
            (ASLStudyLoss()(lp,lq,yy[begin:end]).sum()/17).backward()
            offset += count
        assert offset == len(image_features)
        checks['last_17_studies_16_plus_1_weighted_gradient'] = {'passed': True,
            'gradient_max_abs_error': close(full_grad, split_w.grad),
            'all_images_included': True, 'window_studies': 17, 'microbatch_studies': [16,1]}

        z = torch.randn(3,10).requires_grad_()
        yy = torch.randint(0,2,(3,10)).float()
        lp,lq = -F.softplus(-z),-F.softplus(z)
        expected = ASLStudyLoss()(lp,lq,yy)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            under_autocast = ASLStudyLoss()(lp,lq,yy)
        assert under_autocast.dtype == torch.float32
        before = torch.is_grad_enabled()
        with torch.no_grad():
            assert not ASLStudyLoss()(lp,lq,yy).requires_grad
            assert not torch.is_grad_enabled()
        assert torch.is_grad_enabled() == before
        with torch.inference_mode():
            assert not ASLStudyLoss()(lp,lq,yy).requires_grad
        close(ASLStudyLoss(reduction='mean')(lp,lq,yy), expected.mean())
        close(ASLStudyLoss(reduction='sum')(lp,lq,yy), expected.sum())
        checks['fp32_autocast_reductions_and_grad_context'] = {'passed': True,
            'autocast_max_abs_error': close(under_autocast,expected), 'output_dtype': str(under_autocast.dtype),
            'global_grad_flag_preserved': True}

        rejected = 0
        for args in [(torch.zeros(2,3),torch.zeros(2,2),torch.zeros(2,3)),
                     (torch.zeros(3),torch.zeros(3),torch.zeros(3)),
                     (torch.zeros(0,3),torch.zeros(0,3),torch.zeros(0,3)),
                     (torch.zeros(2,0),torch.zeros(2,0),torch.zeros(2,0))]:
            try:
                ASLStudyLoss()(*args)
            except ValueError:
                rejected += 1
        assert rejected == 4
        checks['invalid_shapes_rejected'] = {'passed': True, 'cases': rejected}
        receipt['status'] = 'passed'
        receipt['passed_checks'] = len(checks)
        receipt['module_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        receipt['status'] = 'failed'
        receipt['error'] = traceback.format_exc()
        raise
    finally:
        (root / 'asl_checks.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'passed_checks': len(checks),
                      'receipt': str(root / 'asl_checks.json')}, ensure_ascii=True))


if __name__ == '__main__':
    _cpu_checks()
