import numpy as np
import torch
torch.manual_seed(0)    
np.random.seed(0)

import pandas as pd
import polars as pl

from ellipsoid import ELLIPSOID
from blackbox import BlackBox

from scipy.spatial.transform import Rotation as ROT
# from torchcontrol.transform import Rotation as R


class Table:
    def __init__(self, blackbox: BlackBox ,hyperparameters: dict, main_variables: dict, dev = torch.device("cpu"), debug=False):
        self.debug = debug
        self.blackbox = blackbox
        self.hyperparameters = hyperparameters
        self.main_variables = main_variables
        self.sphere_resolution = self.main_variables["sphere_resolution"]
        self.device = torch.device(dev)


    def prepare(self):
        # this gets the best model from the BlackBox
        # self.blackbox.prepare_best_model()
        target_center, target_semiaxis, target_rotation, target_geometric_volume = self.blackbox.get_target_params()
        
        # self.blackbox.predict()

        self.blackbox.validate(hyperparameters = self.hyperparameters)

        models = self.blackbox.get_predictions().to(device = self.device)
        # models = self.blackbox.ransac.model.to(device = self.device)
        pred_hyp = models[:, :10].view(len(models), 10, 1)
        pred_semiax = models[:, 13:16].view(len(models), 1, 3)
        pred_evecs = models[:, 16:25].view(len(models), 3, 3)
        pred_cent = models[:, 10:13].view(len(models), 1, 3)


        synth_ellip = ELLIPSOID(
                                params = [target_center, target_semiaxis, target_rotation], 
                                points = None, 
                                fill = True, 
                                resolution=self.sphere_resolution,
                                dev=self.device
                            )

        
        res_new = synth_ellip.get_distance_to(hypothesis=pred_hyp)
        if self.debug: print("res_new", res_new.shape, res_new.dtype)

        

        distances = torch.abs(res_new)
        if self.debug: print("distances", distances.shape, distances.dtype)


        # mm1 = pred_hyp[:, 0] <= 0
        # res_new[mm1.flatten()] *= -1

        if self.debug: print(res_new)


        mask = torch.any((res_new <= 0.0), dim=2)
        if self.debug: print("mask", mask.shape, mask.dtype)
        
        

        intersection = torch.sum(mask, dim=1)
        if self.debug: print("intersect", intersection.shape, intersection.dtype)

        v_pred = (4.0/3.0)*torch.pi*torch.prod(pred_semiax, dim=2).flatten()
        if self.debug: print(v_pred)
        if self.debug: print("v_pred", v_pred.shape, v_pred.dtype)

        v_orig = torch.full(size = (res_new.shape[0],), fill_value=target_geometric_volume, device=self.device)
        # if self.debug: print(v_orig)
        if self.debug: print("v_orig", v_orig.shape, v_orig.dtype)

        min_val = torch.hstack((v_orig.unsqueeze(1), v_pred.unsqueeze(1))).min(dim=1).values    # use the min value between v_prig and v_pred as the intersction 
        if self.debug: print(min_val)
        if self.debug: print("min_val", min_val.shape)
        
        v_intersect =  min_val * intersection / self.sphere_resolution
        if self.debug: print(v_intersect)
        if self.debug: print("v_intersect", v_intersect.shape, v_intersect.dtype)
        
        IoU =  v_intersect / (v_orig + v_pred - v_intersect)
        if self.debug: print("IoU", IoU.shape)

        if self.debug: print(IoU)
        if self.debug: print(torch.max(IoU))


        hyp_a, hyp_b, hyp_c, hyp_d, hyp_e, hyp_f, hyp_g, hyp_h, hyp_i, hyp_j = torch.tensor_split(pred_hyp.flatten(1), 10, dim=-1)

        pred_x, pred_y, pred_z = torch.tensor_split(pred_cent.flatten(1), 3, dim=-1)
        pred_a, pred_b, pred_c = torch.tensor_split(pred_semiax.flatten(1), 3, dim=-1)
        pred_rot_x, pred_rot_y, pred_rot_z, pred_rot_w = torch.tensor_split(torch.from_numpy(ROT.from_matrix(pred_evecs.detach().cpu()).as_quat()).flatten(1), 4, dim=-1)

        real_x, real_y, real_z = target_center
        real_a, real_b, real_c = target_semiaxis
        real_rot_x, real_rot_y, real_rot_z, real_rot_w = ROT.from_matrix(target_rotation).as_quat()


        
        # if self.debug: print("HYP_0", pred_hyp[0])
        # if self.debug: print(hyp_a[0], hyp_b[0], hyp_c[0], hyp_d[0], hyp_e[0], hyp_f[0], hyp_g[0], hyp_h[0], hyp_i[0], hyp_j[0])


        table_dict = {
            "hyp_a": hyp_a.flatten().detach().cpu().tolist(),
            "hyp_b": hyp_b.flatten().detach().cpu().tolist(),
            "hyp_c": hyp_c.flatten().detach().cpu().tolist(),
            "hyp_d": hyp_d.flatten().detach().cpu().tolist(),
            "hyp_e": hyp_e.flatten().detach().cpu().tolist(),
            "hyp_f": hyp_f.flatten().detach().cpu().tolist(),
            "hyp_g": hyp_g.flatten().detach().cpu().tolist(),
            "hyp_h": hyp_h.flatten().detach().cpu().tolist(),
            "hyp_i": hyp_i.flatten().detach().cpu().tolist(),
            "hyp_j": hyp_j.flatten().detach().cpu().tolist(),

            "dataset": np.array([self.blackbox.tomato.dataset] * self.main_variables["n_iterations"]).tolist(),
            "sector": np.array([self.blackbox.tomato.sector] * self.main_variables["n_iterations"]).tolist(),
            "pcd_name": np.array([self.blackbox.tomato.pcd_name] * self.main_variables["n_iterations"]).tolist(),
            "method": np.array([self.hyperparameters["method"]]* self.main_variables["n_iterations"]).tolist(),
            "tomato_id": torch.full(fill_value=self.blackbox.tomato.id, size= (self.main_variables["n_iterations"],)).detach().cpu().tolist(),
            "confine_coeff": np.array([self.hyperparameters["confine_coeff"]] * self.main_variables["n_iterations"]).tolist(),
            "K": torch.full(fill_value=self.blackbox.tomato.confine, size= (self.main_variables["n_iterations"],)).detach().cpu().tolist(),
            "h": np.array([self.hyperparameters["h"]] * self.main_variables["n_iterations"]).tolist(),
            "threshold": np.array([self.hyperparameters["threshold"]] * self.main_variables["n_iterations"]).tolist(),
            "sphere_resolution": np.array([self.sphere_resolution] * self.main_variables["n_iterations"]).tolist(),

            "iou": IoU.detach().cpu().tolist(),
            "intersect_volume": v_intersect.detach().cpu().tolist(),
            "intersect_point_count": intersection.detach().cpu().tolist(),

            "pred_volume": v_pred.detach().cpu().tolist(), 
            "pred_x": pred_x.flatten().detach().cpu().tolist(),
            "pred_y": pred_y.flatten().detach().cpu().tolist(),
            "pred_z": pred_z.flatten().detach().cpu().tolist(),
            "pred_a": pred_a.flatten().detach().cpu().tolist(),
            "pred_b": pred_b.flatten().detach().cpu().tolist(),
            "pred_c": pred_c.flatten().detach().cpu().tolist(),
            "pred_rot_x": pred_rot_x.flatten().detach().cpu().tolist(),
            "pred_rot_y": pred_rot_y.flatten().detach().cpu().tolist(),
            "pred_rot_z": pred_rot_z.flatten().detach().cpu().tolist(),
            "pred_rot_w": pred_rot_w.flatten().detach().cpu().tolist(),

            "real_volume": v_orig.detach().cpu().tolist(), 
            "real_x": torch.full(fill_value=real_x, size= (self.main_variables["n_iterations"],)).tolist(),
            "real_y": torch.full(fill_value=real_y, size= (self.main_variables["n_iterations"],)).tolist(),
            "real_z": torch.full(fill_value=real_z, size= (self.main_variables["n_iterations"],)).tolist(),
            "real_a": torch.full(fill_value=real_a, size= (self.main_variables["n_iterations"],)).tolist(),
            "real_b": torch.full(fill_value=real_b, size= (self.main_variables["n_iterations"],)).tolist(),
            "real_c": torch.full(fill_value=real_c, size= (self.main_variables["n_iterations"],)).tolist(),
            "real_rot_x": torch.full(fill_value=real_rot_x, size= (self.main_variables["n_iterations"],)).tolist(),
            "real_rot_y": torch.full(fill_value=real_rot_y, size= (self.main_variables["n_iterations"],)).tolist(),
            "real_rot_z": torch.full(fill_value=real_rot_z, size= (self.main_variables["n_iterations"],)).tolist(),
            "real_rot_w": torch.full(fill_value=real_rot_w, size= (self.main_variables["n_iterations"],)).tolist(),
            "real_points_count": torch.full(fill_value=self.blackbox.tomato.points_count, size= (self.main_variables["n_iterations"],)).tolist(),
            "water_volume": torch.full(fill_value=self.blackbox.tomato.water_volume, size= (self.main_variables["n_iterations"],)).tolist(),

            "rejected": models[:,-1].detach().cpu().tolist(),
            "inlier_value": self.blackbox.inliers_value.detach().cpu().tolist(),
            "ransac_iteration": list(range(1, self.main_variables["n_iterations"]+1))

        }

        # lens = []
        # for key in table_dict.keys():
        #     lens.append((key, len(table_dict[key])))
        # print("LENS: ", lens)

        # df = pd.DataFrame.from_dict(table_dict, orient='columns')
        # self.set_dtypes(df)

        df = pl.from_dict(table_dict)
        df = self.set_dypes_df(df)

        return df


    def set_dypes_df(self, df):
        return df.cast({
                        "hyp_a": pl.Float64,
                        "hyp_b": pl.Float64,
                        "hyp_c": pl.Float64,
                        "hyp_d": pl.Float64,
                        "hyp_e": pl.Float64,
                        "hyp_f": pl.Float64,
                        "hyp_g": pl.Float64,
                        "hyp_h": pl.Float64,
                        "hyp_i": pl.Float64,
                        "hyp_j": pl.Float64,
                        "pred_volume": pl.Float64,
                        "pred_x": pl.Float64,
                        "pred_y": pl.Float64,
                        "pred_z": pl.Float64,
                        "pred_a": pl.Float64,
                        "pred_b": pl.Float64,
                        "pred_c": pl.Float64,
                        "pred_rot_x": pl.Float64,
                        "pred_rot_y": pl.Float64,
                        "pred_rot_z": pl.Float64,
                        "pred_rot_w": pl.Float64,
                        "real_volume": pl.Float64,
                        "real_x": pl.Float64,
                        "real_y": pl.Float64,
                        "real_z": pl.Float64,
                        "real_a": pl.Float64,
                        "real_b": pl.Float64,
                        "real_c": pl.Float64,
                        "real_rot_x": pl.Float64,
                        "real_rot_y": pl.Float64,
                        "real_rot_z": pl.Float64,
                        "real_rot_w": pl.Float64,
                        "water_volume": pl.Float64,
                        "real_points_count": pl.UInt16,
                        "rejected": pl.UInt8,
                        "inlier_value": pl.Float64,
                        "ransac_iteration": pl.UInt16,
                        "dataset": pl.String,
                        "sector": pl.String,
                        "pcd_name": pl.String,
                        "method": pl.String,
                        "tomato_id": pl.UInt16,
                        "confine_coeff": pl.UInt8,
                        "K": pl.Float64,
                        "h": pl.Float64,
                        "threshold": pl.Float64,
                        "sphere_resolution": pl.UInt16,
                        "iou": pl.Float64,
                        "intersect_volume": pl.Float64,
                        "intersect_point_count": pl.UInt16,
                    })
    
    def set_dtypes(self, df):
        df["hyp_a"] =                               df["hyp_a"].astype('float64')
        df["hyp_b"] =                               df["hyp_b"].astype('float64')
        df["hyp_c"] =                               df["hyp_c"].astype('float64')
        df["hyp_d"] =                               df["hyp_d"].astype('float64')
        df["hyp_e"] =                               df["hyp_e"].astype('float64')
        df["hyp_f"] =                               df["hyp_f"].astype('float64')
        df["hyp_g"] =                               df["hyp_g"].astype('float64')
        df["hyp_h"] =                               df["hyp_h"].astype('float64')
        df["hyp_i"] =                               df["hyp_i"].astype('float64')
        df["hyp_j"] =                               df["hyp_j"].astype('float64')


        df["pred_volume"] =                         df["pred_volume"].astype('float64')
        df["pred_x"] =                              df["pred_x"].astype('float64')
        df["pred_y"] =                              df["pred_y"].astype('float64')
        df["pred_z"] =                              df["pred_z"].astype('float64')
        df["pred_a"] =                              df["pred_a"].astype('float64')
        df["pred_b"] =                              df["pred_b"].astype('float64')
        df["pred_c"] =                              df["pred_c"].astype('float64')
        df["pred_rot_x"] =                          df["pred_rot_x"].astype('float64')
        df["pred_rot_y"] =                          df["pred_rot_y"].astype('float64')
        df["pred_rot_z"] =                          df["pred_rot_z"].astype('float64')
        df["pred_rot_w"] =                          df["pred_rot_w"].astype('float64')


        df["real_volume"] =                         df["real_volume"].astype('float64')
        df["real_x"] =                              df["real_x"].astype('float64')
        df["real_y"] =                              df["real_y"].astype('float64')
        df["real_z"] =                              df["real_z"].astype('float64')
        df["real_a"] =                              df["real_a"].astype('float64')
        df["real_b"] =                              df["real_b"].astype('float64')
        df["real_c"] =                              df["real_c"].astype('float64')
        df["real_rot_x"] =                          df["real_rot_x"].astype('float64')
        df["real_rot_y"] =                          df["real_rot_y"].astype('float64')
        df["real_rot_z"] =                          df["real_rot_z"].astype('float64')
        df["real_rot_w"] =                          df["real_rot_w"].astype('float64')
        df["water_volume"] =                        df["water_volume"].astype('float64')
        df["real_points_count"] =                   df["real_points_count"].astype('int16')

        df["rejected"] =                            df["rejected"].astype('int8')
        df["inlier_value"] =                        df["inlier_value"].astype('int16')
        df["ransac_iteration"] =                    df["ransac_iteration"].astype('int16')

        df["dataset"] =                             df["dataset"].astype('object')
        df["sector"] =                              df["sector"].astype('object')
        df["pcd_name"] =                            df["pcd_name"].astype('object')
        df["method"] =                              df["method"].astype('object')
        df["tomato_id"] =                           df["tomato_id"].astype('int16')
        df["confine_coeff"] =                       df["confine_coeff"].astype('int8')
        df["K"] =                                   df["K"].astype('float64')
        df["h"] =                                   df["h"].astype('float64')
        df["threshold"] =                           df["threshold"].astype('float64')
        df["sphere_resolution"] =                   df["sphere_resolution"].astype('int16')
        df["iou"] =                                 df["iou"].astype('float64')
        df["intersect_volume"] =                    df["intersect_volume"].astype('float64')
        df["intersect_point_count"] =               df["intersect_point_count"].astype('int16')

