from toolbox import *
import torch

from blackbox import BlackBox
from table import Table
from read_write import WRITER, READER
# from buffer import PRQ

from tqdm.auto import tqdm

class LargeTable:
    def __init__(self, list_of_files: list, scale_correction: dict, main_variables: dict, dev: str = "cpu", dataset_details: dict = {}):
        self.list_of_files = list_of_files
        self.scale_correction = scale_correction
        self.main_variables = main_variables
        self.main_records = []
        self.device = torch.device(dev)
        self.dataset_details = dataset_details
        self.dataset = self.dataset_details["dataset"]
        self.sector = self.dataset_details["sector"]


        self.writer = WRITER(dataset = self.dataset, sector=self.sector)
        self.reader = READER(dataset = self.dataset, sector=self.sector)
        
        ### Filtering the list of the files
        self.list_of_files = [file_addr for file_addr in list_of_files if ".npz" in file_addr]
        self.file_count = len(self.list_of_files)
        print(f"Total of {len(self.list_of_files)} files.")
        # print(*self.list_of_files[:3], sep="\n")


    def get_table(self, threshold, confine_coeff, h, method, blackbox, remove_rejctions=False):
        hyper = {
                    "threshold": threshold,
                    "confine_coeff": confine_coeff,
                    "h": h,
                    "method": method,
                }

        df = Table(hyperparameters=hyper, main_variables=self.main_variables, blackbox=blackbox, debug=False, dev= self.device).prepare()

        if remove_rejctions:
            df = df.dropna()
            df.drop(df[df['rejected'] > 0].index, inplace = True)

        return df


    def prepare(self, remove_rejctions: bool =True, print_resolution: int = 5, composed_hyperparams = [(0.1, 3, 3, "count")]):
        count = 0
        ii = 0
        for file_addr in tqdm(self.list_of_files, desc=" File", position=0):
            blackbox = BlackBox(address = file_addr, main_variables=self.main_variables, scale_correction= self.scale_correction, dev= self.device)

            ii += 1
            # if (ii) % print_resolution == 0: print(f"-> {ii}/{self.file_count}", sep=', ')
            c = 0
            for threshold, confine_coeff, h, method in tqdm(composed_hyperparams, desc=" Param Combination", position=1, leave=False):
                c += 1
                self.write(df=self.get_table(
                                threshold=threshold, 
                                confine_coeff=confine_coeff, 
                                h=h, method=method, 
                                blackbox=blackbox,
                                remove_rejctions = remove_rejctions,
                                ),
                                end = (ii == len(self.list_of_files) and c == len(composed_hyperparams)))

    def write(self, df, end=False):
        self.writer.write(df, end=end)      

    def load(self, columns, filter_predicate):
        return self.reader.read(columns = columns, filter_predicate=filter_predicate)

    async def sink(self, each_n_file):
        await self.read_write.sink(each_n_file=each_n_file)
