import torch
torch.manual_seed(0)    


from ransac import RANSAC
from ellipsoid import ELLIPSOID
from tomato import Tomato

import os

class BlackBox:
    def __init__(self, address: str, scale_correction: tuple = (100, 100, 100), main_variables: dict = {}, dev = torch.device("cpu"), debug=False):
        self.debug = debug
        self.raw_address = address
        self.tomato = Tomato(address=address, scale_correction=scale_correction)
        self.best_model = None
        self.pred_file_address = None

        n_iterations = main_variables["n_iterations"]
        self.sphere_resolution = main_variables["sphere_resolution"]

        ###### Get Predictions on the raw points
        self.figure = ELLIPSOID(params = [], points = self.tomato.points, resolution = self.sphere_resolution, debug=False, dev = dev) 
        self.ransac = RANSAC(
                            figure = self.figure,
                            n_iterations = n_iterations,
                            debug=debug,
                            dev = dev,
                            pred_filename= self.raw_address.replace("raw_points", "predictions")
                            )
        
        self.fit()


    def fit(self):
        self.ransac.fit()

    def validate(self, hyperparameters):
        self.ransac.validate(
                            params= [
                                    hyperparameters["confine_coeff"] * self.tomato.confine, 
                                    hyperparameters["h"]
                                    ]
                            )
        

        self.inliers_value = self.ransac.count_inliers(
                                                method= hyperparameters["method"], 
                                                threshold= hyperparameters["threshold"]
                                                )
        
    def prepare_best_model(self):
        # self.predict()
        models = self.get_predictions()

        mask = (models[:,-1] == 0)
        inds = torch.nonzero(mask)
        if self.debug: print(inds.shape)

        inliers = self.inliers_value[inds]
        max_inlier_ind = torch.argmax(inliers).item()
        max_inlier_ind_from_nonfiltered = inds[max_inlier_ind]

        # best model
        self.best_model = models[max_inlier_ind_from_nonfiltered].flatten()
        if self.debug: print(self.best_model)
        


    def get_target_params(self):
        return [self.tomato.center, self.tomato.semiaxis, self.tomato.rotation, self.tomato.geometric_volume]


    def get_predictions(self):
        return self.ransac.get_model()

