import os
import xml.etree.ElementTree as ET


def split_list_into_n_parts(lst, n):
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def main(base_route, task_num, algo):
    route_root = os.path.dirname(base_route)
    route_name = os.path.basename(base_route).split('.xml')[0]

    os.makedirs(f'{route_root}/{algo}/', exist_ok=True)

    tree = ET.parse(base_route)
    root = tree.getroot()
    case = root.findall('route')
    results = split_list_into_n_parts(case, task_num)
    for index, re in enumerate(results):
        new_root = ET.Element("routes")
        for x in re:
            new_root.append(x)
        new_tree = ET.ElementTree(new_root)
        new_tree.write(f'{route_root}/{algo}/{route_name}_{index}.xml', encoding='utf-8', xml_declaration=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_route", default='bench2drive/leaderboard/data/bench2drive220.xml', type=str)
    parser.add_argument("--task_num", default=8, type=int)
    parser.add_argument("--algo", default='splits8', type=str)
    args = parser.parse_args()

    main(args.base_route, args.task_num, args.algo)