import json
import glob
import argparse
import os

# Default root for auto-merge (--all mode)
# script lives at bench2drive/tools/, eval dir is two levels up at Bench2Drive/evaluation/
_EVAL_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../evaluation'
)

def merge_route_json(folder_path, quiet=False):
    file_paths = sorted(glob.glob(f'{folder_path}/*.json'))
    merged_records = []
    driving_score = []
    success_num = 0
    crashed_total = 0
    for file_path in file_paths:
        if 'merged.json' in file_path: continue
        try:
            with open(file_path) as file:
                data = json.load(file)
        except (json.JSONDecodeError, KeyError):
            print(f'  [skip] {os.path.basename(file_path)} (empty or invalid JSON)')
            continue
        try:
            records = data['_checkpoint']['records']
        except (KeyError, TypeError):
            print(f'  [skip] {os.path.basename(file_path)} (missing _checkpoint/records)')
            continue
        # Remove "Failed - Simulation crashed" records from source file in-place
        clean_records = [rd for rd in records if rd.get('status') != 'Failed - Simulation crashed']
        crashed = len(records) - len(clean_records)
        if crashed > 0:
            crashed_total += crashed
            data['_checkpoint']['records'] = clean_records
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            if not quiet:
                print(f'  [cleaned] {os.path.basename(file_path)}: removed {crashed} crashed route(s)')
        for rd in clean_records:
            rd.pop('index', None)
            merged_records.append(rd)
            driving_score.append(rd['scores']['score_composed'])
            if rd['status']=='Completed' or rd['status']=='Perfect':
                success_flag = True
                for k,v in rd['infractions'].items():
                    if len(v)>0 and k != 'min_speed_infractions':
                        success_flag = False
                        break
                if success_flag:
                    success_num += 1
                    if not quiet:
                        print(rd['route_id'])
    if crashed_total > 0 and quiet:
        print(f'  [cleaned] removed {crashed_total} crashed route(s) from source files')
    if len(merged_records) != 220:
        print(f"-----------------------Warning: there are {len(merged_records)} routes in your json, which does not equal to 220. All metrics (Driving Score, Success Rate, Ability) are inaccurate!!!")
    merged_records = sorted(merged_records, key=lambda d: d['route_id'], reverse=True)
    _checkpoint = {
        "records": merged_records
    }

    n = len(driving_score)
    ds = sum(driving_score) / n if n else 0.0
    sr = success_num / n if n else 0.0

    merged_data = {
        "_checkpoint": _checkpoint,
        "driving score": ds,
        "success rate": sr,
        "eval num": n,
    }

    with open(os.path.join(folder_path, 'merged.json'), 'w') as file:
        json.dump(merged_data, file, indent=4)

    return n, ds, sr


def merge_all(eval_root):
    eval_root = os.path.realpath(eval_root)
    subdirs = sorted([
        d for d in os.listdir(eval_root)
        if os.path.isdir(os.path.join(eval_root, d))
    ])

    if not subdirs:
        print(f'No subdirectories found under {eval_root}')
        return

    results = []
    for name in subdirs:
        folder = os.path.join(eval_root, name)
        # skip folders with no non-merged json files
        jsons = [f for f in glob.glob(f'{folder}/*.json') if 'merged.json' not in f]
        if not jsons:
            continue
        print(f'\n[merge] {name}')
        n, ds, sr = merge_route_json(folder, quiet=True)
        if n == 0:
            print(f'  [skip] no valid records found')
            continue
        results.append((name, n, ds, sr))

    # summary table
    col = max(len(r[0]) for r in results) if results else 10
    header = f"{'folder':<{col}}  {'routes':>6}  {'driving_score':>13}  {'success_rate':>12}"
    print('\n' + '=' * len(header))
    print(header)
    print('-' * len(header))
    for name, n, ds, sr in results:
        print(f'{name:<{col}}  {n:>6}  {ds:>13.4f}  {sr*100:>11.2f}%')
    print('=' * len(header))
    print(f'Total: {len(results)} experiments merged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge per-GPU route JSON files.')
    parser.add_argument('-f', '--folder', default=None,
                        help='Merge a single folder. If omitted, merges all subdirs under --eval_root.')
    parser.add_argument('--eval_root', default=_EVAL_ROOT,
                        help=f'Root evaluation directory for auto-merge (default: {_EVAL_ROOT})')
    args = parser.parse_args()

    if args.folder:
        n, ds, sr = merge_route_json(args.folder)
        print(f'\nDriving Score: {ds:.4f}  Success Rate: {sr*100:.2f}%  ({n} routes)')
    else:
        merge_all(args.eval_root)
