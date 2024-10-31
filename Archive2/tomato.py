import numpy as np
import json

class Tomato:
    def __init__(self, address: str, scale_correction: tuple = (100, 100, 100)):
        self.address = address
        self.loaded = False
        self.scale_correction_vol, self.scale_correction_ax, self.scale_correction_conf  = scale_correction

        try:
            if ".npz" in self.address:
                self.data = np.load(self.address)
                self.loaded = True
        except Exception as ex:
            self.loaded = False

        if self.loaded:
            self.points = self.data["points"]
            self.points_count = self.data["points_count"].item()
            self.id = int(self.data["id"].item())
            self.sector = self.data["sector"].item()
            self.dataset = self.data["dataset"].item()
            self.pcd_name = self.data["pcd_name"].item()
            self.water_volume = self.data["water_volume"].item()
            self.geometric_volume = self.data["geometric_volume"].item() * self.scale_correction_vol
            self.center = self.data["center"]
            self.semiaxis = self.data["semiaxis"] * self.scale_correction_ax
            self.rotation = self.data["rotation"]
            self.confine = self.data["confine"] * self.scale_correction_conf

            if self.dataset == "wood":
                self.quick_fix_wood()

    def quick_fix_wood(self):
        with open(f"wood_gt/{self.sector}.json", 'r') as f:
            json_data = json.load(f)


            keys = list(json_data.keys())
            centers = np.array([np.array(json_data[keys[i]]["center"])for i, k in enumerate(keys)])
            centers -= self.center
            # centers -= self.points.mean(axis=0)
            dists = np.linalg.norm(centers, axis=1)
            # print("DIST", dists)
            
            nearest_i = dists.argmin()

            self.id = int(keys[nearest_i])
            id_str = str(keys[nearest_i])
            tm = json_data[id_str]

            self.water_volume = tm["volume"]
            self.geometric_volume = tm["geometric_volume"] * self.scale_correction_vol
            self.center = np.array(tm["center"])
            self.semiaxis = np.array(tm["semiaxis"]) * self.scale_correction_ax
            self.rotation = np.array(tm["rotation"])


            theta = np.radians(21.5)
            Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]
                ])
            
            self.center = np.dot(self.center, Rx.T)
            # self.points = np.dot(self.points, Rx.T)

            