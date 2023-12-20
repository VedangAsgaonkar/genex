
def get_dataset(split, hps):
    if hps.dataset == 'vec':
        from .vec import Dataset
        dataset = Dataset(hps.dfile, split, hps.batch_size)
    elif hps.dataset == 'ts':
        from .ts import Dataset
        dataset = Dataset(hps.dfile, split, hps.batch_size, hps.time_steps)
    else:
        raise Exception()

    assert dataset.d == hps.dimension

    return dataset
