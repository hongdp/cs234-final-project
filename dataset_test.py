import dataset as ds
import config as config

def test_0():
    dataset = ds.WarfarinDataSet(config.ClinicalDataSetConfig())
    for col in dataset:
        print(col)

if __name__ == '__main__':
    test_0()
