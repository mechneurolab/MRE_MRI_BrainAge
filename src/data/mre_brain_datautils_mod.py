#%%
import nibabel as nib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm

#%%
def common_path(arr, pos='prefix'):
    # The longest common prefix of an empty array is "".
    if not arr:
        print("Longest common", pos, ":", "")
    # The longest common prefix of an array containing 
    # only one element is that element itself.
    elif len(arr) == 1:
        print("Longest common", pos, ":", str(arr[0]))
    else:
        dir = range(len(arr[0])) if pos=="prefix" else range(-1,-len(arr[0])+1,-1)
        # Sort the array
        arr.sort()
        result = ""
        # Compare the first and the last string character
        # by character.
        for i in dir:
            #  If the characters match, append the character to
            #  the result.
            if arr[0][i] == arr[-1][i]:
                result += arr[0][i]
            # Else, stop the comparison
            else:
                break
    if pos=="suffix":
        result = result[::-1]
    print("Longest common", pos, ":", result)
    return result

def read_files(data_folder_path, label_folder_path, set_id, only_map=False):
    labels = pd.read_csv(label_folder_path+'labels_final.csv')
    labels_list = []
    map_list = []
    sex_list = []
    study_list = []
    meta_list = []
    for root, dirs, files in os.walk(data_folder_path):
        common_prefix = common_path(files, pos="prefix")
        common_suffix = common_path(files, pos="suffix")
        for id in set_id:
            age =  labels.loc[labels["ID"] == id,'Age'].to_numpy()[0]
            sex =  labels.loc[labels["ID"] == id,'Antipodal_Sex'].to_numpy()[0]
            study = labels.loc[labels["ID"] == id,'Study_ID'].to_numpy()[0]
            filename = common_prefix + str(id) + common_suffix
            if not os.path.exists(root+filename):
                filename = common_prefix + "{:0>3d}".format(id) + common_suffix
            nib_raw = nib.load(data_folder_path + filename)
            meta = nib_raw.header
            map = nib_raw.get_fdata()[:,:,:]
            labels_list.append(age)
            sex_list.append(sex)
            map_list.append(map)
            study_list.append(study)
            meta_list.append(meta)
    X_map = np.array(map_list).astype(np.float32)
    X_sex = np.array(sex_list)
    X_study = np.array(study_list)
    y = np.array(labels_list).astype(np.float32)
    m = np.array(meta_list)
    if only_map:
        output = X_map
    else:
        output = (X_map, X_sex, X_study, y, m)
    return output

def get_bin_centers(bin_range, bin_step):
    """
    Compute bin centers
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)
    return bin_centers


def distribute_target(x, bin_centers, sigma=1):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_step = int(bin_centers[1]-bin_centers[0])
    bin_start = bin_centers[0] - float(bin_step) / 2
    # bin_end = bin_centers[0] + float(bin_step) / 2
    # bin_length = bin_end - bin_start
    bin_number = len(bin_centers)
    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v

def preprocess(X_train, X_test, X_val=None, preproc_type='std'):
    # X_train_pp, X_test_pp = np.zeros_like(X_train), np.zeros_like(X_test)
    X_train_ = X_train.copy()
    X_test_ = X_test.copy()
    if type(X_val)==np.ndarray:
        X_val_ = X_val.copy()
        

    if preproc_type == 'constant':
        X_train_pp = np.nan_to_num(X_train_,nan=-1e-3)
        X_test_pp = np.nan_to_num(X_test_,nan=-1e-3)
        if type(X_val)==np.ndarray:
            X_val_pp = np.nan_to_num(X_val_,nan=-1e-3)

    elif preproc_type == 'std':
        mu = np.nanmean(X_train_)
        sigma = np.nanstd(X_train_)
        X_train_[np.isnan(X_train_)] = mu
        X_train_pp = (X_train_-mu)/sigma
        # mu_adj = np.nanmean(X_train_pp)
        # X_train_pp[np.isnan(X_train_pp)] = mu_adj
        X_test_[np.isnan(X_test_)] = mu
        X_test_pp = (X_test_-mu)/sigma
        # X_test_pp[np.isnan(X_test_pp)] = mu_adj

        if type(X_val)==np.ndarray:
            X_val_[np.isnan(X_val_)] = mu
            X_val_pp = (X_val_-mu)/sigma
            # X_val_pp[np.isnan(X_val_pp)] = mu_adj


    if type(X_val)==np.ndarray:
        out = (X_train_pp, X_test_pp, X_val_pp)
    else:
        out = (X_train_pp, X_test_pp)

    return out

def data_loading(map_type, ids, path='/work/cniel/sw/BrainAge/datasets/', only_map=False):
    # Define the map type
    # assert (map_type=='Stiffness' or map_type=='DR' or map_type=='Volume' or 
            # map_type=='Stiffness-Volume' or map_type=='Stiffness-DR' or map_type=='Volume-DR' or 
            # map_type=='Stiffness-Volume-DR')
    train_id, val_id, test_id = ids
    # Load brain maps
    folder_path_input = path+'{map}_FINAL/'.format(map=map_type) # maps path
    folder_path_labels = path # labels path
    print('Loading training set for {map} maps...'.format(map=map_type))
    X_train_map, X_train_sex, X_train_study, y_train, m_train = read_files(folder_path_input, folder_path_labels, train_id)
    print('Loading validation set for {map} maps...'.format(map=map_type))
    X_val_map, X_val_sex, X_val_study, y_val, m_val = read_files(folder_path_input, folder_path_labels, val_id)
    print('Loading test set for {map} maps...'.format(map=map_type))
    X_test_map, X_test_sex, X_test_study, y_test, m_test = read_files(folder_path_input, folder_path_labels, test_id)
    # Preprocessing map 
    X_train_pp, X_test_pp, X_val_pp = X_train_map, X_test_map, X_val_map
    # One hot encoding for categorical variables
    # define one hot encoding
    if not only_map:
        encoder = OneHotEncoder(sparse_output=False)
        # transform categorical variables
        X_train_sex = encoder.fit_transform(X_train_sex.reshape(-1,1))
        X_train_study = encoder.fit_transform(X_train_study.reshape(-1,1))
        X_train_cat = np.concatenate((X_train_sex,X_train_study), axis=1)
        X_val_sex = encoder.fit_transform(X_val_sex.reshape(-1,1))
        X_val_study = encoder.fit_transform(X_val_study.reshape(-1,1))
        X_val_cat = np.concatenate((X_val_sex,X_val_study), axis=1)
        X_test_sex = encoder.fit_transform(X_test_sex.reshape(-1,1))
        X_test_study = encoder.fit_transform(X_test_study.reshape(-1,1))
        X_test_cat = np.concatenate((X_test_sex,X_test_study), axis=1)
        # Arranging data for CNN input 
        train_data = [X_train_pp, X_train_cat]
        train_target = y_train
        val_data = [X_val_pp, X_val_cat]
        val_target = y_val
        test_data = [X_test_pp, X_test_cat]
        test_target = y_test
        output = (train_data, train_target, val_data, val_target, test_data, test_target)
    else:
        train_data = X_train_pp
        val_data = X_val_pp
        test_data = X_test_pp
        output = (train_data, val_data, test_data)

    return output

def masking(mask,data):
    for i in range(len(mask)):
        mask_subject = np.isnan(mask[i])
        data_masked = data.copy()
        data_masked[i][mask_subject] = np.nan
    return data_masked

def data_splitting(map_type, ids, path='/work/cniel/sw/BrainAge/datasets/'):
    train_data_, val_data_, test_data_ = data_loading('Stiffness', ids, only_map=True, path=path)
    mask_train_data = train_data_.copy()
    mask_val_data = val_data_.copy()
    mask_test_data = test_data_.copy()
    # train_data_raw_list = []
    # val_data_raw_list = []
    # test_data_raw_list = []
    # train_data_pp_list = []
    # val_data_pp_list = []
    # test_data_pp_list = []
    if map_type == 'Stiffness' or map_type == 'Volume' or map_type =='DR':
        map_types = map_type.split('-')
        train_data, train_target, val_data, val_target, test_data, test_target = data_loading(map_type, ids, path=path)
        # print(val_data[0].shape)
        train_data_raw = masking(mask_train_data, train_data[0])
        val_data_raw = masking(mask_val_data, val_data[0])
        test_data_raw = masking(mask_test_data, test_data[0])
        # print(val_data_raw.shape)
        # print(val_data[-1].shape)
        train_data_pp, test_data_pp, val_data_pp  = preprocess(train_data_raw, test_data_raw, val_data_raw, preproc_type='std')
        # Arrange list of maps
        train_data_raw_list = [train_data_raw]
        val_data_raw_list = [val_data_raw]
        test_data_raw_list = [test_data_raw]
        train_data_pp_list = [train_data_pp, train_data[-1]]
        val_data_pp_list = [val_data_pp, val_data[-1]]
        test_data_pp_list = [test_data_pp, test_data[-1]]
        # print(val_data_pp.shape)
        # print(val_data[-1].shape)
        # Fit with the entire dataset.
        # X_train_pp_all = np.concatenate((train_data[0], val_data[0]))
        # X_train_cat_all = np.concatenate((train_data[-1], val_data[-1]))
        # train_data_all = [X_train_pp_all, X_train_cat_all]
        # if map_type == 'Volume' or map_type =='DR':
        #     print(f'Loading {map_type} raw data')
        #     train_data_raw, train_target_raw, val_data_raw, val_target_raw, test_data_raw, test_target_raw = data_loading(map_type, ids, path=path)
        #     mask_train_data = masking(mask_train_data, train_data_raw)
        #     mask_val_data = masking(mask_val_data, val_data_raw)
        #     mask_test_data = masking(mask_test_data, test_data_raw)
    elif map_type == 'Stiffness-Volume' or map_type == 'Stiffness-DR' or map_type == 'Volume-DR':
        map_types = map_type.split('-')

        train_data, train_target, val_data, val_target, test_data, test_target = data_loading(map_types[0], ids, path=path)
        train_data_raw = masking(mask_train_data, train_data[0])
        val_data_raw = masking(mask_val_data, val_data[0])
        test_data_raw = masking(mask_test_data, test_data[0])
        train_data_pp, test_data_pp, val_data_pp = preprocess(train_data_raw, test_data_raw, val_data_raw, preproc_type='std')
        
        train_data_1, val_data_1, test_data_1 = data_loading(map_types[1], ids, only_map=True, path=path)
        train_data_1_raw = masking(mask_train_data, train_data_1)
        val_data_1_raw = masking(mask_val_data, val_data_1)
        test_data_1_raw = masking(mask_test_data, test_data_1)
        train_data_1_pp, test_data_1_pp, val_data_1_pp  = preprocess(train_data_1_raw, test_data_1_raw, val_data_1_raw, preproc_type='std')
        # Arrange list of maps
        train_data_raw_list = [train_data_raw, train_data_1_raw]
        val_data_raw_list = [val_data_raw, val_data_1_raw]
        test_data_raw_list = [test_data_raw, test_data_1_raw]
        train_data_pp_list = [train_data_pp, train_data_1_pp, train_data[-1]]
        val_data_pp_list = [val_data_pp, val_data_1_pp, val_data[-1]]
        test_data_pp_list = [test_data_pp, test_data_1_pp, test_data[-1]]
        # train_data.insert(1, train_data_1)
        # val_data.insert(1, val_data_1)
        # test_data.insert(1, test_data_1)
        # # Fit with the entire dataset.
        # X_train_pp_all = np.concatenate((train_data[0], val_data[0]))
        # X_train_pp_all_1 = np.concatenate((train_data[1], val_data[1]))
        # X_train_cat_all = np.concatenate((train_data[-1], val_data[-1]))
        # train_data_all = [X_train_pp_all, X_train_pp_all_1, X_train_cat_all]
    elif map_type == 'Stiffness-Volume-DR':
        map_types = map_type.split('-')

        train_data, train_target, val_data, val_target, test_data, test_target = data_loading(map_types[0], ids, path=path)
        train_data_raw = masking(mask_train_data, train_data[0])
        val_data_raw = masking(mask_val_data, val_data[0])
        test_data_raw = masking(mask_test_data, test_data[0])
        train_data_pp, test_data_pp, val_data_pp  = preprocess(train_data_raw, test_data_raw, val_data_raw, preproc_type='std')

        train_data_1, val_data_1, test_data_1 = data_loading(map_types[1], ids, only_map=True, path=path)
        train_data_1_raw = masking(mask_train_data, train_data_1)
        val_data_1_raw = masking(mask_val_data, val_data_1)
        test_data_1_raw = masking(mask_test_data, test_data_1)
        train_data_1_pp, test_data_1_pp, val_data_1_pp  = preprocess(train_data_1_raw, test_data_1_raw, val_data_1_raw, preproc_type='std')

        train_data_2, val_data_2, test_data_2 = data_loading(map_types[2], ids, only_map=True, path=path)
        train_data_2_raw = masking(mask_train_data, train_data_2)
        val_data_2_raw = masking(mask_val_data, val_data_2)
        test_data_2_raw = masking(mask_test_data, test_data_2)
        train_data_2_pp, test_data_2_pp, val_data_2_pp  = preprocess(train_data_2_raw, test_data_2_raw, val_data_2_raw, preproc_type='std')

        # Arrange list of maps
        train_data_raw_list = [train_data_raw, train_data_1_raw, train_data_2_raw]
        val_data_raw_list = [val_data_raw, val_data_1_raw, val_data_2_raw]
        test_data_raw_list = [test_data_raw, test_data_1_raw, test_data_2_raw]
        train_data_pp_list = [train_data_pp, train_data_1_pp, train_data_2_pp, train_data[-1]]
        val_data_pp_list = [val_data_pp, val_data_1_pp, val_data_2_pp, val_data[-1]]
        test_data_pp_list = [test_data_pp, test_data_1_pp, test_data_2_pp, test_data[-1]]

        # train_data.insert(1, train_data_1)
        # val_data.insert(1, val_data_1)
        # test_data.insert(1, test_data_1)
        # train_data.insert(2, train_data_2)
        # val_data.insert(2, val_data_2)
        # test_data.insert(2, test_data_2)
        # # Fit with the entire dataset.
        # X_train_pp_all = np.concatenate((train_data[0], val_data[0]))
        # X_train_pp_all_1 = np.concatenate((train_data[1], val_data[1]))
        # X_train_pp_all_2 = np.concatenate((train_data[2], val_data[2]))
        # X_train_cat_all = np.concatenate((train_data[-1], val_data[-1]))
        # train_data_all = [X_train_pp_all, X_train_pp_all_1, X_train_pp_all_2, X_train_cat_all]
    # y_train_all = np.concatenate((train_target, val_target))
    # train_target_all = y_train_all
    # get_bin_centers(bin_range, bin_step)
    data_dict = {
        'mask_train_data':mask_train_data,
        'mask_val_data':mask_val_data,
        'mask_test_data':mask_test_data,
        'train_data': train_data_pp_list, 
        'train_target': train_target,
        'val_data': val_data_pp_list,
        'val_target': val_target,
        'test_data': test_data_pp_list,
        'test_target': test_target,
        'train_data_raw': train_data_raw_list, 
        'val_data_raw': val_data_raw_list,
        'test_data_raw': test_data_raw_list,
    }
    return data_dict