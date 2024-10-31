import os

def list_files_recursive(path='.', my_list = []):
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path, my_list)
        else:
            my_list.append(full_path)

def compose_hyperparams(hyperparameters: dict):
    composed_hyperparams = []
    for method in hyperparameters["method"]:
        for threshold in hyperparameters["threshold"]:
                for confine_coeff in hyperparameters["confine_coeff"]:
                    for h in hyperparameters["h"]:
                            composed_hyperparams.append((threshold, confine_coeff, h, method))

    return composed_hyperparams




import json
from parameters import *
def plot_manip(plt, data):
    dataset_details = data["dataset_details"]
    x_label = data["x_label"]
    y_label = data["y_label"]
    tt = y_label + "_" + x_label 
    tt = tt.replace(" ", "_")


    json_dirr = os.path.join("tablesdata", dataset_details["dataset"])
    if not os.path.isdir(json_dirr):
         os.makedirs(json_dirr)       

    fig_dirr = os.path.join("figures", dataset_details["dataset"])
    if not os.path.isdir(fig_dirr):
         os.makedirs(fig_dirr)   

         
    with open(f'{json_dirr}/{dataset_details["sector"]}_{plot_method}_{plot_threshold}_{tt}.json', 'w') as f:
        json.dump(data["save_dict"], f)

        # Add title and labels
    
    plt.title(f'{x_label} vs {y_label}. {dataset_details["dataset"]}_{dataset_details["sector"]}, method: {plot_method}, thresh: {plot_threshold}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    

    plt.grid(which='both', axis='x')
    plt.legend()

    # Show the plot
    plt.savefig(f'{fig_dirr}/{dataset_details["sector"]}_{plot_method}_{plot_threshold}_{tt}.pdf',
                # dpi=600, 
                # bbox_inches='tight'
                )


