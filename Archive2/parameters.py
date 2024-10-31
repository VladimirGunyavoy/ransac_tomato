# volume, semiaxis, confine
scale_corrections = {
"polygon": (1e-6, 1, 100),
"wood": (1e-6, 1., 100.),
"synthetic": (1, 1., 1.)
}

main_variables = {
    "sphere_resolution": 5000,
    "n_iterations": 10000
}

hyperparameters1 = {
    "threshold": [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5],
    "confine_coeff": [3, 5],
    "h": [2, 3],
    "method": ["count", 
            #    "median", "average", "sum"
               ],
}

hyperparameters2 = {
    "threshold": [0],
    "confine_coeff": [100],
    "h": [100],
    "method": [
                # "count", 
               "median", "average", "sum"
               ],
}



color_list = ["black","dimgrey","indianred","red","chocolate","orange"
,"gold"
,"olive"
,"yellowgreen"
,"lightgreen"
,"seagreen"
,"turquoise"
,"powderblue"
,"deepskyblue"
,"steelblue"
,"royalblue"
,"navy"
,"blueviolet"
,"violet"
,"magenta"
,"hotpink"
,"pink"]





plot_method = "count"
plot_threshold = 0.1
plot_confine_coeff = 5
plot_h = 3