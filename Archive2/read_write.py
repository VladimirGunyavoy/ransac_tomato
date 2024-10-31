import polars as pl
import duckdb as db
import os 

class WRITER:
    def __init__(self, dataset, sector):
        self.dataset = dataset
        self.sector = sector
        self.counter = 0
        self.chunksize = 43 * 20
        self.container = []
        self.len_container = 0
        # self.datapath = Path(f"./database/{self.dataset}/{self.sector}/")
        self.datapath = f"database/{self.dataset}/{self.sector}/"
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)

        self.num_cores = os.cpu_count()

    def write(self, dict_arr, end = False):
        if not end:
            if self.len_container + 1 < self.chunksize:
                self.container.append(dict_arr)
                self.len_container += 1
            else:
                self.container.append(dict_arr)
                self.__save()
                
        else:

            self.container.append(dict_arr)
            self.__save()

        
    def __save(self):
        pl.concat(self.container).write_parquet(
            file = f"database/{self.dataset}/{self.sector}/{self.sector}_{self.counter}.parquet",
            # compression_level = 21,
        )
        self.counter += 1
        self.container = []
        self.len_container = 0




class READER:
    def __init__(self, dataset, sector):
        self.dataset = dataset
        self.sector = sector

        # self.datapath = Path(f"./database/{self.dataset}/{self.sector}/")
        self.datapath = f"database/{self.dataset}/{self.sector}/*.parquet"

        self.num_cores = os.cpu_count()

    def read(self, columns, filter_predicate):
        con = db.connect()

        # db.scan_parquet(self.datapath)
        cl = columns if len(columns) else "*"

        query = f""" SELECT {cl}
        FROM read_parquet("{self.datapath}")
        """ 

        if len(filter_predicate):
            query += " WHERE " + filter_predicate

        df = db.query(query).pl()

        con.close()

        return df
                
