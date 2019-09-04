import numpy as np

def partitions(number, k):
    '''
    Distribution of the folds
    Args:
        number: number of patients
        k: folds number
    '''
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[1:(number % k)] += 1
    return n_partitions

def get_indices(n_splits = 3, subjects = 40):
    '''
    Indices of the set test
    Args:
        n_splits: folds number
        subjects: number of patients
    '''
    l = partitions(subjects, n_splits)
    fold_sizes = l
    indices = np.arange(subjects).astype(int)
    current = 1
    for fold_size in fold_sizes:
        start = current
        stop =  current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])

def get_cv_path(img_idx):
    '''
    Get the path of patient
    Args:
        img_idx: gives the index of images
    '''
    imgs_path = []

    for i in (img_idx):
        if i < 10:
            i = '0' + str(i)
            imgs_path.append("Patient_" + i)
        else:
            i = str(i)
            imgs_path.append("Patient_" + i)
    return imgs_path

def k_folds(n_splits = 3, subjects = 40):
    '''
    Generates folds for cross validation
    Args:
        n_splits: folds number
        subjects: number of patients
    '''
    indices = np.arange(subjects).astype(int) 
    for test_idx in get_indices(n_splits, subjects):
        train_idx = np.setdiff1d(indices, test_idx)
        train_idx = np.setdiff1d(train_idx, 0)

        train_imgs = get_cv_path(train_idx)
        test_imgs = get_cv_path(test_idx)
        yield train_imgs, test_imgs


if __name__ == "__main__":
    for train_idx, test_idx in k_folds(n_splits = 2, subjects = 3):
        print("train_idx = {} ".format(train_idx))
        print("")
        print("test_idx = {}".format(test_idx))
