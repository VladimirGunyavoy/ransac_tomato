import torch
import numpy as np
torch.manual_seed(0)    
np.random.seed(0)

class ELLIPSOID:
    def __init__(self, params: list = [], resolution: int = 1500, points = None, fill: bool = False, dev = torch.device("cpu"), debug=False):
        self.debug = debug
        self.resolution = resolution
        self.dimension = 3
        self.fill = fill
        self.eps = 1e-10
        self.device = torch.device(dev)

        """ params: [center, semiaxis_vector, rotation_matrix]"""
        self.params = params
        
        self.points = points if points is not None else self.generate_points()
        
        if self.points.shape[-1] != self.dimension:
            raise RuntimeError(f"Your points tensor should be of shape m x n x {self.dimension}.")

        if self.debug: print("ELLIPSOID CONSTRUCT")
        

    def generate_uniform_points(self):
        # Generate uniform azimuthal angle (phi)
        phi = np.random.uniform(0, 2 * np.pi, self.resolution)
        
        # Generate polar angle (theta) with correct distribution
        # theta should be distributed as cos(theta) ~ uniform(-1, 1)
        theta = np.arccos(2 * np.random.uniform(0, 1, self.resolution) - 1)
        
        if not self.fill:
            u = np.ones((theta.shape))
        else:
            u = np.random.uniform(0., 1., size = theta.shape)
        
        r = (u)**(1./3.)
        # Convert to Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        return np.column_stack((x, y, z))

    def deform_points(self, uniform_points):
        points = np.dot(uniform_points, np.diag(self.params[1]))
        points = points @ np.array(self.params[2]).T
        points += np.array(self.params[0])
        return points

    def generate_points(self):
        """Deform a Sphere"""
        """ params: [center, semiaxis_vector, rotation_matrix]"""

        points = self.generate_uniform_points()
            

        if len(self.params) != 0:
            points = self.deform_points(points)
            # raise RuntimeError(f"You should define params: [center, semiaxis_vector, rotation_matrix].")
        else:
            if self.debug: print("Creating a Sphere as no params are provided...!")
            if self.debug: print("params: [center, semiaxis_array, rotation_matrix]")

        return points
    
    def prepare_columns(self, samples = None):
        if samples is None:
            samples = self.points
        x,y,z = torch.tensor_split(torch.from_numpy(samples), self.dimension, dim=-1)
        return torch.hstack((x**2, y**2, z**2, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z)).to(dtype=torch.float32)
    
    def build_model(self, quadric):
        """
        matrix = self.build_matrix(quadric)  
        center = np.linalg.solve(matrix["Fa"], matrix["Fb"].T)
        translation_matrix = np.eye(4)
        translation_matrix[3, :3] = center.T
        R1 = translation_matrix.dot(matrix["mk"]).dot(translation_matrix.T)
        evals, evecs = np.linalg.eig(R1[:3, :3] / -R1[3, 3])
        evecs = evecs.T
        radii = np.sqrt(1. / np.abs(evals))
        radii *= np.sign(evals)
        """

        # quadric = quadric.to(device = torch.device("cpu"))
        
        W = quadric.clone().to(device = self.device)

        # ### For the sign change
        # mask = (W[:, 0] * W[:, -1]) < 0
        # W[mask, -1] *= -1

        # if W[:,0] < 0:
        
        # if self.debug: print("BEFORE", W[0])

        # mmask = W[:,0] < 0
        # W[mmask, :] *= -1

        # if self.debug: print("AFTER", W[0])


        # W[:, 3:6] /= 2.0

        # A, B, C, D, E, F, G, H, I, J
        Q = W[:, [0, 3, 4, 3, 1, 5, 4, 5, 2]].view(len(W), self.dimension, self.dimension)


        L = 2 * W[:, [6, 7, 8]].view(len(W), self.dimension, 1)

        if self.debug: print("Q", Q.shape)
        if self.debug: print("L", L.shape)

        
        try:
            Q_inv = torch.linalg.inv(Q)
        except RuntimeError:
            if self.debug: print("Warning: Matrix Q is singular or close to singular. Using pseudoinverse.")
            Q_inv = torch.linalg.pinv(Q)

        center = (torch.transpose(-L, 1, 2) @ Q_inv)/2 #-1

        
        if self.debug: print("q_inv", Q_inv.shape)
        if self.debug: print("center", center.shape)
        

        TP = 0.25 * (torch.transpose(L, 1, 2) @ Q_inv @ L ) + 1
        if self.debug: print("TP", TP.shape)
        
        
        try:
            vals, evecs = torch.linalg.eigh(Q)
        except RuntimeError:
            if self.debug: print("Warning: Eigenvalue computation failed. Using SVD as fallback.")
            U, S, Vt = torch.linalg.svd(Q)
            vals, evecs = S, U


        if self.debug: print("vals", vals.shape)
        if self.debug: print("evecs", evecs.shape)
        
        
        radii = torch.sqrt(torch.divide(TP, vals.unsqueeze(1)))
        radii = torch.nan_to_num(radii, nan=self.eps) # replace nan with zero -> later we will face error in "(np.max(semiaxes, axis=1)/np.min(semiaxes, axis=1)) > h"
        if self.debug: print("radii",radii.shape)
        
        return torch.hstack((W, center.flatten(1), radii.flatten(1), evecs.flatten(1), torch.zeros(len(quadric), 1, device=self.device)))


    def get_distance_to(self, hypothesis):
        if len(hypothesis.shape) != 3:
            raise RuntimeError(f"The Given hypotheis should be of shape m * n * 3")
        if hypothesis.shape[-1] != 1:
            raise RuntimeError(f"The Given hypotheis should be of shape {len(hypothesis)} * {self.dimension**2 + 1} * 1")
        
        pt = self.prepare_columns(samples=None).to(device=self.device) # shape: n_iterations x sphere_resolution x 9
        pt = torch.hstack((pt, torch.full((pt.shape[0], 1), fill_value = 1,  device=self.device)))

        # # ### For the sign change
        # mask = (pt[:, 0] * pt[:, -1]) < 0
        # pt[mask, -1] *= -1

        # if self.debug: print("BEFORE", pt[0])

        # mmask = pt[:,0] < 0
        # pt[mmask, :] *= -1

        # if self.debug: print("AFTER", pt[0])

        #### No need for repeat
        # pt = pt.repeat(len(hypothesis), 1, 1)# shape: n_iterations x sphere_resolution x 10

        
        return pt @ hypothesis
