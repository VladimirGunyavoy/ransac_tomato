import torch
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
torch.manual_seed(0)    
np.random.seed(0)

import os

from quadric import QUADRIC

class RANSAC:
    def __init__(self, figure, n_iterations: int = 10, dev = torch.device("cpu"), debug=False, pred_filename=""):
        self.debug = debug
        self.n_iterations = n_iterations
        self.quadric = QUADRIC(figure = figure, n_samples=self.n_iterations, debug=False)
        self.device = torch.device(dev)
        self.model = None
        self.pred_file_address = pred_filename
        self.loaded = False
        if self.debug: print("RNSK CONSTRUCT")


    def __pred_file_exists(self):
        # return os.path.isfile(self.pred_file_address)
        return False
        
    def fit(self):
        if not self.loaded:
            if not self.__pred_file_exists():
                self.__proceed_with_predictions()
                self.__save_preds()
                self.loaded = False
            else:
                self.model = torch.from_numpy(np.load(self.pred_file_address)['data']).to(device=self.device)
                self.loaded = True

        
    
    def __proceed_with_predictions(self):
        self.model = self.quadric.get_model()
        
        
    def __save_preds(self):
        pred_dir = r'predictions/synthetic/normal_noise/'
        fine_name = self.pred_file_address.split('\\')[-1]
        # pred_dir = os.path.join(*(self.pred_file_address.split('\\')[:-1]))
        # pred_dir = r'/'.join(pred_dir)

        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        # pred_dir += self.pred_file_address.split('\\')[-2:]
        # pred_dir += r'/'.join(self.pred_file_address.split('\\')[-2:])
        np.savez_compressed(pred_dir+fine_name, data=self.model.detach().cpu())
        

    def get_model(self):
        # if self.loaded:
            return self.model
        # else:
        #     raise RuntimeError("The model is not loaded!")
    
    def validate(self, params: list = []):
        if len(params) == 0:
            raise Exception(f"You should provide params: [k, h]")
        k, h = params

        semiaxes = self.model[:, 13:16].detach().cpu().numpy()
        if self.debug: print("SHAPE", semiaxes.shape)
        
        condition_1 = np.any(np.isnan(semiaxes), axis=1)
        if self.debug: print("condition_1", condition_1.shape)
        # if self.debug: print(condition_1)
        
        condition_2 = np.any(semiaxes < self.quadric.figure.eps, axis=1)
        if self.debug: print("condition_2", condition_2.shape)
        # if self.debug: print(condition_2)

        condition_3 = np.any(semiaxes > k, axis=1)
        if self.debug: print("condition_3", condition_3.shape)
        # if self.debug: print(condition_3)

        condition_4 = (np.max(semiaxes, axis=1)/np.min(semiaxes, axis=1)) > h
        if self.debug: print("condition_4", condition_4.shape)
        # if self.debug: print(condition_4)

        condition = torch.hstack((torch.from_numpy(condition_1).unsqueeze(1), torch.from_numpy(condition_2).unsqueeze(1), torch.from_numpy(condition_3).unsqueeze(1), torch.from_numpy(condition_4).unsqueeze(1)))
        if self.debug: print("condition_5", condition.shape)

        condition = torch.sum(condition, axis=1)
        if self.debug: print("condition_6", condition.shape)

        self.model[:, -1] = condition
            

    def count_inliers(self, method, threshold):
        res = self.quadric.figure.get_distance_to(hypothesis=self.model[:, :10].view(len(self.model), 10, 1)).to(device=self.device, dtype=torch.float32)
        if self.debug: print("RES", res.shape)
        # if self.debug: print(res[0][:20])

        if (method == "median"):
            # mask = torch.median(res) <= threshold
            # internal = torch.count_nonzero(mask, dim=1).flatten()
            internal = torch.median(res, dim=1).values.flatten()
            

        elif (method == "count"):
            mask = torch.abs(res) <= threshold
            internal = torch.count_nonzero(mask, dim=1).flatten()

        elif (method == "average"):
            # mask = torch.mean(res) <= threshold
            # internal = torch.count_nonzero(mask, dim=1).flatten()
            internal = torch.mean(res, dim=1).flatten()

        elif (method == "sum"):
            # mask = torch.sum(res) <= threshold
            # internal = torch.count_nonzero(mask, dim=1).flatten()
            internal = torch.sum(res, dim=1).flatten()
        
        else:
            ## Default on "count"
            mask = torch.abs(res) <= threshold
            internal = torch.count_nonzero(mask, dim=1).flatten()

        return internal



    
        