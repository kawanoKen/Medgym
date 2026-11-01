#!/usr/bin/env python3
"""Evaluate the six released case-study policies and export Figure 4 rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_ind_cluster_pop as evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, default=ROOT / 'checkpoints-case-pinn')
    parser.add_argument('--save-dir', type=Path, default=ROOT / 'results/online/eval/case-study')
    parser.add_argument('--n-eval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init-noise-std', type=float, default=0.0)
    args = parser.parse_args()
    if args.n_eval < 1 or args.init_noise_std < 0:
        parser.error('n-eval must be positive and init-noise-std must be nonnegative')
    case = args.case_dir.resolve()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    results = evaluation._worker(
        202347, 'lagrangian_trpo', str(case / 'pinn/individual'),
        str(case / 'online/individual'), str(case / 'online/population'),
        str(case / 'pinn/population/scales.npy'), str(case / 'online/cluster_pooled'),
        str(case / 'pinn/cluster_pooled'), {202347: 1}, args.n_eval, 20,
        args.init_noise_std, args.seed,
    )
    if results is None or any(value is None for value in results.values()):
        raise RuntimeError('All six case-study policies and patient PINN are required')
    args.save_dir.mkdir(parents=True, exist_ok=True)
    evaluation._save_trajectories(202347, results, str(args.save_dir), 'lagrangian_trpo')
    metrics = {key: {name: float(result[name]) for name in (
        'mean_sofa_all', 'mean_sofa', 'final_sofa', 'mean_lac', 'safety_rate'
    )} for key, result in results.items()}
    manifest = {
        'script': 'scripts/online/eval_case_study.py', 'patient_id': 202347,
        'K': 20, 'horizon_hours': 96, 'dt_min': 0.5, 'dt_max': 36,
        'fixed_dt': 4.8, 'n_eval': args.n_eval, 'seed': args.seed,
        'init_noise_std': args.init_noise_std,
        'case_dir': str(case), 'state_and_action_conversion': True,
        'run_definition': 'Repeated evaluation episodes of one released checkpoint per policy; not independent training seeds',
        'metric_sampling': 'Existing evaluator: post-step SOFA/lactate held from interval start on hourly grid',
        'trajectory_sampling': 'States at interval ends; applied actions on [start, end); no linear interpolation',
        'sha256': {str(p.relative_to(case)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(case.rglob('*')) if p.is_file() and p.suffix in ('.pt', '.npy', '.npz')
                   and not p.name.startswith('._') and 'eval' not in p.parts},
        'metrics': metrics,
    }
    from data.config import device
    manifest['device'] = str(device)
    (args.save_dir / '_meta.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Rollouts and evaluation metrics saved to {args.save_dir}')


if __name__ == '__main__':
    main()
