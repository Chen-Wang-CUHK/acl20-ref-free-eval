import os
import csv
import argparse


def collect_eval_results(opts):
    log_dir = opts.log_folder
    saved_dir = opts.saved_folder
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    log_files = []
    if os.path.isdir(log_dir):
        for sub_dir in ['tac_08', 'tac_09', 'tac_2010', 'tac_2011']:
            sub_log_dir = os.path.join(log_dir, sub_dir)
            log_files = log_files + [f for f in os.listdir(sub_log_dir)]
    elif os.path.isfile(log_dir):
        log_files = [log_dir]
    else:
        print("The log_dir is invalid: {}".format(log_dir))
        raise NotImplementedError

    log_files = sorted(log_files)

    saved_csv = os.path.join(saved_dir, "{}_collected.csv".format(opts.csv_name_base))
    assert not os.path.isfile(saved_csv), "The {} already exists! Make sure your saved_folder is correctly configured.".format(saved_csv)
    print("Evaluation results are saved in {}".format(saved_csv))

    fieldnames = ['model_name',
                  'tac_2011_pearson_r', 'tac_2011_spearman_rho', 'tac_2011_kendall_tau', 'blank1',
                  'tac_2010_pearson_r', 'tac_2010_spearman_rho', 'tac_2010_kendall_tau', 'blank2',
                  'tac_09_pearson_r', 'tac_09_spearman_rho', 'tac_09_kendall_tau', 'blank3',
                  'tac_08_pearson_r', 'tac_08_spearman_rho', 'tac_08_kendall_tau']

    with open(saved_csv, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        rslt_dict = {}
        for f in log_files:
            one_rslt_dict = {}
            sub_dir = f.split('_')[0] + '_' + f.split('_')[1]
            log_file = open(os.path.join(log_dir, sub_dir, f))
            print("Preprocessing {}".format(os.path.join(log_dir, f)))

            model_name = f[len(sub_dir)+1:-len('_log.txt')]
            if model_name not in rslt_dict:
                rslt_dict[model_name] = {}
                for k in fieldnames:
                    rslt_dict[model_name][k] = ''
                rslt_dict[model_name]['model_name'] = model_name

            for line in log_file:
                line = line.strip()

                for corr_metric in ['pearson_r', 'spearman_rho', 'kendall_tau']:
                    if corr_metric in line and 'mean' in line and 'p_'+corr_metric+':' not in line:
                        r_list = line.split(':')[1].strip().split(',')
                        r_list = [r.strip() for r in r_list]
                        max_r = r_list[0].split()[-1]
                        min_r = r_list[1].split()[-1]
                        mean_r = r_list[2].split()[-1]
                        median_r = r_list[3].split()[-1]
                        rslt_dict[model_name][sub_dir + '_' + corr_metric] = mean_r
                        continue
        for model_name in rslt_dict:
            writer.writerow(rslt_dict[model_name])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="collect_evaluation_results.py")
    parser.add_argument("--log_folder", "-log_folder", type=str, default='logs',
                        help="The folder that stored the evaluation logs.")
    parser.add_argument("--saved_folder", "-saved_folder", type=str, default='logs',
                        help="The folder to save the collected evaluation results.")
    parser.add_argument("--csv_name_base", "-csv_name_base", type=str, default='all_eval_results',
                        help="The name base of the saved .csv file.")
    opts = parser.parse_args()

    collect_eval_results(opts)