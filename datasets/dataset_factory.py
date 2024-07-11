from .scannetpp import ScanNetppDataset

def get_dataset(cfg, name):
    if cfg.data.category == "scannetpp":
        return ScanNetppDataset(cfg, name)
    else:
        print("Dataset name incorrect!")