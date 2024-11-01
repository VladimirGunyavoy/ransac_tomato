import torch
torch.manual_seed(0)    


class QUADRIC:
    def __init__(self, figure, n_samples: int = 1, dev = torch.device("cpu"), debug=False):
        self.debug = debug
        self.figure = figure
        self.n_samples = n_samples
        self.sample_size = self.figure.dimension**2
        self.device = torch.device(dev)

        if self.debug: print("QUADRIC CONSTRUCT")

    def __sample(self):
        if len(self.figure.points) < self.sample_size:
            raise Exception(f"should have at least {self.sample_size} points!")
        
        random_tensor = torch.rand(self.n_samples, len(self.figure.points))
        permutations = random_tensor.argsort(dim=1)
        return self.figure.points[permutations[:, :self.sample_size]]


    def get_hypothesis(self):
        samples = self.__sample()

        eq_sys = self.figure.prepare_columns(samples).view(len(samples), self.sample_size, self.sample_size).transpose(2,1).to(dtype=torch.float32, 
                                                                                                                               device = self.device
                                                                                                                               )
        eq_ans = torch.ones(len(samples), self.sample_size, 1, dtype=torch.float32, 
                            device = self.device
                            )
        try:
            sys_sol = torch.linalg.solve(eq_sys, eq_ans).flatten(1)
        except Exception as e:
            sys_sol = torch.zeros(len(samples), self.sample_size, dtype=torch.float32, 
                            device = self.device
                            )
# fill_value = -1/1
        return torch.hstack((sys_sol, torch.full(size = (len(sys_sol), 1), fill_value=1))).to(device = self.device, dtype=torch.float32)
    
    def get_model(self):
        return self.figure.build_model(quadric = self.get_hypothesis())
