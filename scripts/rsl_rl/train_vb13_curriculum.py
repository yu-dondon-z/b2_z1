"""One VB13 stage: guarded training, then actor-only validation, never blind promotion."""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from train_vb12_guarded import read_series, stop_owned

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('stages', ROOT / 'source/b2_z1/b2_z1/tasks/manager_based/b2_z1/mdp/vb13_curriculum.py')
stages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stages)


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check_previous(path, checkpoint, stage):
    certificate = json.loads(Path(path).read_text())
    if (certificate.get('protocol_version') != 1 or certificate.get('stage') != stage-1
            or certificate.get('passed') is not True
            or Path(certificate.get('checkpoint', '')).resolve() != checkpoint.resolve()
            or certificate.get('sha256') != checksum(checkpoint)):
        raise ValueError('Previous stage has not passed validation for this exact checkpoint')


def guard_reason(series, stage, warmup=200, window=50):
    for tag, samples in series.items():
        if any(not math.isfinite(v) for _, v in samples):
            return f'Nonfinite scalar: {tag}'
    ready = series.get('VB13/hold_ready_fraction', [])
    if len(ready) < max(warmup, window):
        return None
    mean = lambda tag: sum(v for _, v in series.get(tag, [])[-window:])/window
    contacts = 'Episode_Termination/gun_contact_budget'
    if len(series.get(contacts, [])) >= window and mean(contacts) > .85 and mean('VB13/hold_ready_fraction') < .01:
        return 'Preparation collapsed into persistent gun support/contact'
    if stage == 4 and mean('VB12/requested_motion_fraction') < .05 and mean('VB11/startup_ready_fraction') < .05:
        return 'Final motion stage cannot complete measured preparation'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=range(5), default=0)
    parser.add_argument('--iterations', type=int, default=500, help='NEW iterations for this stage')
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'logs/rsl_rl/b2_z1_amp/2026-10-13_10-35-50_VB10_clearance_fromVB9_10999/model_12300.pt')
    parser.add_argument('--previous-acceptance', type=Path)
    args = parser.parse_args()
    ckpt = args.checkpoint.expanduser().resolve()
    if min(args.iterations, args.num_envs) <= 0 or not ckpt.is_file():
        parser.error('Need positive sizes and an existing checkpoint')
    if args.stage > 0:
        if not args.previous_acceptance:
            parser.error('Stage >0 requires --previous-acceptance from the previous stage')
        try:
            check_previous(args.previous_acceptance, ckpt, args.stage)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
    if ckpt.parent.parent.resolve() != (ROOT/'logs/rsl_rl/b2_z1_amp').resolve():
        parser.error('Checkpoint must be inside this repo logs/rsl_rl/b2_z1_amp/<run>/')
    stage = stages.stage_spec(args.stage)
    audit = ROOT/'outputs'/f'vb13_stage{args.stage}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
    audit.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(B2_Z1_VB13_STAGE=str(args.stage), B2_Z1_HOSE_MASS_KG=str(stage['hose_kg']),
        PYTHONUNBUFFERED='1', HYDRA_FULL_ERROR='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    # Older playback ancestors only accept the legacy override [1,10]kg.
    # VB13 configuration applies its authoritative stage load AFTER them,
    # including zero-hose stage 0. Do not feed 0 into the legacy parser.
    environment.pop('B2_Z1_HOSE_MASS_KG', None)
    command = [sys.executable, str(ROOT/'scripts/rsl_rl/train_amp.py'), '--task', 'Unitree-b2z1-AMP-VB13-Prepare',
        '--headless', '--num_envs', str(args.num_envs), '--max_iterations', str(args.iterations),
        '--resume', '--reset_optimizer', '--load_run', ckpt.parent.name, '--checkpoint', ckpt.name,
        '--run_name', f'VB13_stage{args.stage}_{stage["name"]}', 'agent.amp_cfg.amp_num_preload_transitions=20000']
    print(f'[VB13] Stage {args.stage}: {stage}; NEW iterations={args.iterations}\n[VB13] Logs: {audit}', flush=True)
    child, directory, accumulator, reason = None, None, None, None
    certificate = {'protocol_version': 1, 'stage': args.stage, 'passed': False, 'tests': []}
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        log_path = audit/'training.log'
        with log_path.open('w') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                if directory is None:
                    match = re.search(r'source code is saved to\s+([^\n]+?)/source.tar.gz', log_path.read_text(errors='replace'))
                    if match:
                        directory = Path(match.group(1).strip())
                        accumulator = EventAccumulator(str(directory), size_guidance={'scalars': 0})
                if accumulator is not None:
                    data = read_series(accumulator)
                    reason = guard_reason(data, args.stage)
                    if reason:
                        stop_owned(child)
                        break
                    samples = data.get('VB13/hold_ready_fraction', [])
                    if samples:
                        print(f'[VB13] iteration={samples[-1][0]} measured_hold_ready={samples[-1][1]:.3f}', flush=True)
                time.sleep(20)
        if directory is None:
            match = re.search(r'source code is saved to\s+([^\n]+?)/source.tar.gz', log_path.read_text(errors='replace'))
            if match:
                directory = Path(match.group(1).strip())
                accumulator = EventAccumulator(str(directory), size_guidance={'scalars': 0})
        if reason:
            raise RuntimeError(reason)
        if child.returncode:
            raise RuntimeError(f'Training failed ({child.returncode}); see {log_path}')
        if accumulator is None:
            raise RuntimeError('No training event directory; cannot validate completion')
        data = read_series(accumulator)
        reason = reason or guard_reason(data, args.stage)
        if reason or not data.get('VB13/hold_ready_fraction'):
            raise RuntimeError(reason or 'Missing measured preparation telemetry')
        checkpoint = max(directory.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
        certificate.update(checkpoint=str(checkpoint), sha256=checksum(checkpoint), training_dir=str(directory))
        # Separate GPU jobs: training must finish before deterministic tests.
        for seed in (42, 43):
            trace = audit/f'cold_seed{seed}.csv'
            playback = [sys.executable, str(ROOT/'scripts/rsl_rl/play_amp.py'), '--task', 'Unitree-b2z1-AMP-VB13-Prepare-Play',
                '--checkpoint', str(checkpoint), '--num_envs', '1', '--headless', '--seed', str(seed),
                '--max-steps', '3000' if stage['moving'] else '1500', '--motion-trace-csv', str(trace),
                'env.commands.base_velocity.debug_vis=false', 'env.commands.ee_target.debug_vis=false']
            print(f'[VB13] Actor-only cold validation seed={seed}; not applying the reference teacher', flush=True)
            with (audit/f'cold_seed{seed}.log').open('w') as stream:
                child = subprocess.Popen(playback, cwd=ROOT, env=environment, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                child.wait()
            if child.returncode or not trace.is_file():
                raise RuntimeError(f'Validation execution failed; see cold_seed{seed}.log')
            passed, metrics = stages.trace_acceptance(list(csv.DictReader(trace.open(newline=''))), stage['moving'])
            certificate['tests'].append({'seed': seed, 'passed': passed, **metrics})
            print(f'[VB13] Validation: {passed}, {metrics}', flush=True)
            if not passed:
                reason = f'Cold preparation validation failed (seed {seed}); not advancing'
                break
        certificate['passed'] = len(certificate['tests']) == 2 and all(t['passed'] for t in certificate['tests'])
    except KeyboardInterrupt:
        reason = 'Operator interrupted this stage'
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
    finally:
        if child is not None:
            stop_owned(child)
        certificate['stop_reason'] = reason
        (audit/'acceptance.json').write_text(json.dumps(certificate, indent=2))
        (audit/'status.json').write_text(json.dumps({'command': command, 'training_dir': str(directory),
            'stop_reason': reason, 'passed': certificate['passed']}, indent=2))
    if not certificate['passed']:
        raise SystemExit(f'[VB13] Stage not qualified. Weights/logs retained: {audit}; {reason}')
    print(f'[VB13] Stage {args.stage} passed cold-start validation: {audit}/acceptance.json', flush=True)
    if args.stage < 4:
        print(f'Next stage (NOT automatically started):\n'
              f'bash scripts/rsl_rl/train_amp_vb13_prepare.sh --stage {args.stage+1} '
              f'--checkpoint {certificate["checkpoint"]} --previous-acceptance {audit}/acceptance.json', flush=True)
    else:
        print('[VB13] Next: full RGB-D vision validation, not real deployment approval.', flush=True)


if __name__ == '__main__':
    main()
