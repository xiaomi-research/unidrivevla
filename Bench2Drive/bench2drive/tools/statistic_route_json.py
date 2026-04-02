import os
import copy
import json
import glob
import argparse
import numpy as np
from prettytable import PrettyTable


def is_success(record):
    success_flag = False
    if record['status'] in ['Completed', 'Perfect']:
        success_flag = True
        for k, v in record['infractions'].items():
            if len(v) > 0 and k != 'min_speed_infractions':
                success_flag = False
                break
    return success_flag

def draw_table(route_ids, scenario_names, driving_scores, success_routes):
    table = PrettyTable()
    table.field_names = ["index", "route_id", "scenario_names", "driving_score", "success"]
    for i in range(len(driving_scores)):
        table.add_row([i, route_ids[i], scenario_names[i], driving_scores[i], success_routes[i]])
    return table


def statistic_route_json(route_dir, remove_update=False):
    if remove_update:
        print("WARNING: it will remove and update the failed route file and record !!!!")

    route_paths = glob.glob(f'{route_dir}/*.json')
    route_paths.sort()

    route_ids = []
    town_names = []
    scenario_names = []

    logging_infos = []
    driving_scores = []
    success_routes = []

    total_completed_routes = 0

    for route_path in route_paths:
        if 'merged' in os.path.basename(route_path):
            continue

        with open(route_path) as file:
            data = json.load(file)
            data_checkpoint = data['_checkpoint']

            records = data_checkpoint['records']
            progress = data_checkpoint['progress']
            global_record = data_checkpoint['global_record']

            # finish all clips
            if len(global_record):
                completed_routes = 0
                for record in records:
                    route_ids.append(record['route_id'].split("_")[1])
                    town_names.append(record['town_name'])
                    scenario_names.append(record['scenario_name'])
                    driving_scores.append(record['scores']['score_composed'])
                    if is_success(record):
                        completed_routes += 1
                        total_completed_routes += 1
                        success_routes.append(1)
                    else:
                        success_routes.append(0)

                logging_info = "loading {}, success:{}/{}, progress:{}/{} ".format(
                    os.path.basename(route_path), completed_routes, progress[1], progress[0], progress[1])
                logging_infos.append(logging_info)
            else:
                valid_records = []
                completed_routes = 0
                for record in records:
                    if record['status'] in ['Completed', 'Perfect', 'Failed - TickRuntime', 'Failed - Agent got blocked']:
                        route_ids.append(record['route_id'].split("_")[1])
                        town_names.append(record['town_name'])
                        scenario_names.append(record['scenario_name'])
                        valid_records.append(record)
                        driving_scores.append(record['scores']['score_composed'])
                        if is_success(record):
                            completed_routes += 1
                            total_completed_routes += 1
                            success_routes.append(1)
                        else:
                            success_routes.append(0)
                    else:
                        failed_paths = glob.glob(os.path.join(route_path, '*' + record['save_name']))
                        if remove_update and len(failed_paths):
                            print("this record failed".format())
                            failed_path = failed_paths[0]
                            if os.path.exists(failed_path):
                                if 'meta' in os.listdir(failed_path):
                                    os.system('rm -r {}'.format(failed_path))

                valid_progress = [len(valid_records), progress[1]]

                updated_checkpoint = {
                    'global_record': {},
                    'progress': valid_progress,
                    'records': valid_records,
                }

                update_data = copy.deepcopy(data)
                update_data['_checkpoint'] = updated_checkpoint

                if remove_update:
                    print("update json file: {}".format(route_path))
                    with open(os.path.join(route_path), 'w') as file:
                        json.dump(update_data, file, indent=4)

                logging_info ="loading {}, success:{}/{}, progress:{}/{}".format(
                    os.path.basename(route_path), completed_routes, valid_progress[0], valid_progress[0], valid_progress[1])
                logging_infos.append(logging_info)

    driving_score = np.average(driving_scores)
    success_score = total_completed_routes / (len(driving_scores) + 1e-5) * 100

    # print
    table = draw_table(route_ids, scenario_names, driving_scores, success_routes)
    print(table)
    print()

    for logging_info in logging_infos:
        print(logging_info)
    print()

    print("completed_routes:{}/{}, driving_score:{:.2f}, success_score:{:.2f}".format(
        total_completed_routes, len(driving_scores), driving_score, success_score))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--route_dir', required=True, help='path to evaluation output directory')
    args = parser.parse_args()
    statistic_route_json(args.route_dir)
