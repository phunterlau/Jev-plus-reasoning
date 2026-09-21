"""Verify bundled hashes and metrics offline, using only the standard library."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'archive/source/src'))
from jev_reasoning.metrics import metrics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(folder):
    return [json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]


def main():
    imported = json.loads((ROOT/'provenance/import_manifest.json').read_text())
    for name, expected in imported['sha256'].items():
        assert digest(ROOT/name) == expected, f'Import hash mismatch: {name}'
    r1 = ROOT/'results/r1-mode-capacity-v1'
    manifest = json.loads((r1/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for name, expected in manifest['artifacts'].items():
        assert digest(r1/name) == expected, f'R1 artifact hash mismatch: {name}'
    predictions = rows(r1)
    assert len(predictions) == 144
    assert len({(r['id'], r['size'], r['mode']) for r in predictions}) == 144
    published = json.loads((r1/'small_cost_quality.json').read_text())
    assert published['source_predictions_sha256'] == digest(r1/'predictions.jsonl')
    report = {'r1_small': {}, 'examples': {}}
    for mode in ['nonthinking', 'thinking']:
        subset = [r for r in predictions if r['size']=='small' and r['mode']==mode]
        assert len(subset) == 36 and all(r['cue'] is not None for r in subset)
        adapted = [dict(gold=r['gold'], prediction=r['cue']['prediction'],
            correct=r['cue']['prediction']==r['gold'], readout_valid=True,
            probabilities=r['cue']['probabilities'], option_ids=['True','False','Unknown'],
            total_seconds=r['total_seconds'], sampled_reasoning_tokens=len(r['trace_ids']),
            natural_close=mode=='thinking' and r['closed'], forced_close=False,
            peak_allocated_bytes=r['peak_allocated_bytes']) for r in subset]
        computed = metrics(adapted)
        for key, expected in published['metrics'][mode].items():
            if isinstance(expected, (int, float)):
                assert math.isclose(computed[key], expected, rel_tol=1e-10, abs_tol=1e-10), key
            else:
                assert computed[key] == expected, key
        report['r1_small'][mode] = computed
    small = {m: {r['id']: r['cue']['prediction']==r['gold'] for r in predictions if r['size']=='small' and r['mode']==m} for m in ['nonthinking','thinking']}
    assert small['nonthinking'].keys() == small['thinking'].keys()
    repairs = sum(not small['nonthinking'][i] and small['thinking'][i] for i in small['thinking'])
    regressions = sum(small['nonthinking'][i] and not small['thinking'][i] for i in small['thinking'])
    assert (repairs, regressions) == (23, 0)
    report['paired'] = {'repairs': repairs, 'regressions': regressions}
    for name in ['counting-rasperry-v1', 'counting-strawberry-v1', 'rae-short-v1']:
        folder = ROOT/'results'/name
        m = json.loads((folder/'manifest.json').read_text())
        assert m['status']=='complete' and digest(folder/'predictions.jsonl')==m['predictions_sha256']
        rs = rows(folder)
        assert len(rs)==4
        assert [(r['mode'],r['seed']) for r in rs]==[('nonthinking',17),('thinking',17),('thinking',18),('thinking',19)]
        assert len(m['telemetry'])==5
        for sample in m['telemetry']:
            lines = sample['matching_process'] if isinstance(sample, dict) else sample.splitlines()
            assert any(line.split(',')[0].strip()==str(m['pid']) for line in lines)
        compact=[]
        for r in rs:
            assert r['correct']==(r['prediction']==r['gold'])
            if r['probabilities'] is not None:
                assert len(r['probabilities']) == len(r['options'])
                assert all(math.isfinite(p) and 0<=p<=1 for p in r['probabilities'])
                assert abs(sum(r['probabilities'])-1)<1e-6
                assert r['prediction']==r['options'][max(range(len(r['options'])),key=r['probabilities'].__getitem__)]
                p_gold=r['probabilities'][r['options'].index(r['gold'])]
            else:
                assert r['prediction'] is None and r['natural_close'] is False
                p_gold=None
            compact.append(dict(mode=r['mode'], seed=r['seed'], prediction=r['prediction'],
                correct=r['correct'], p_gold=p_gold, seconds=r['total_seconds'], tokens=len(r['trace_ids'])))
        report['examples'][name]=compact
    assert digest(ROOT/'scripts/run_showcase.py')==json.loads((ROOT/'results/rae-short-v1/manifest.json').read_text())['source_sha256']
    print(json.dumps({'status':'passed','imported_files':len(imported['sha256']),'r1_records':144,'individual_example_records':12,'report':report},indent=2))


if __name__ == '__main__':
    main()
