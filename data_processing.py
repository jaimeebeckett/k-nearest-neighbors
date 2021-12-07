from dataset import Data
import numpy as np


def load_car():
    car_file = 'data/raw/car.data'
    car_data = Data('car', car_file, 'class', names=['buying', 'maint', 'doors', 'persons',
                                                     'lug_boot', 'safety', 'class'])
    car_column_order = {'buying': {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
                        'maint': {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
                        'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
                        'persons': {'2': 0, '4': 1, 'more': 3},
                        'lug_boot': {'small': 0, 'low': 0, 'med': 1, 'high': 2, 'big': 2},
                        'safety': {'small': 0, 'low': 0, 'med': 1, 'high': 2, 'big': 2}}
    car_data.ordinal(car_column_order)
    return car_data


def load_cancer():
    cancer_file = 'data/raw/breast-cancer-wisconsin.data'
    cancer_data = Data('cancer', cancer_file, 'class',
                       names=['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size',
                              'Uniformity of Cell Shape', 'Marginal Adhesion',
                              'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                              'Normal Nucleoli', 'Mitoses', 'class'],
                       usecols=['Clump Thickness', 'Uniformity of Cell Size',
                                'Uniformity of Cell Shape', 'Marginal Adhesion',
                                'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                                'Normal Nucleoli', 'Mitoses', 'class'],
                       na_values='?')
    cancer_data.imputation()
    return cancer_data


def load_vote():
    vote_file = 'data/raw/house-votes-84.data'
    vote_data = Data('vote', vote_file, 'class',
                     names=['class', 'handicapped-infants', 'water-project-cost-sharing',
                            'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                            'el-salvador-aid', 'religious-groups-in-schools',
                            'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile',
                            'immigration', 'synfuels-corporation-cutback', 'education-spending',
                            'superfund-right-to-sue', 'crime', 'duty-free-exports',
                            'export-administration-act-south-africa'],
                     na_values='?')
    vote_data.imputation()
    vote_data.nominal('handicapped-infants')
    vote_data.nominal('water-project-cost-sharing')
    vote_data.nominal('adoption-of-the-budget-resolution')
    vote_data.nominal('physician-fee-freeze')
    vote_data.nominal('el-salvador-aid')
    vote_data.nominal('religious-groups-in-schools')
    vote_data.nominal('anti-satellite-test-ban')
    vote_data.nominal('aid-to-nicaraguan-contras')
    vote_data.nominal('mx-missile')
    vote_data.nominal('immigration')
    vote_data.nominal('synfuels-corporation-cutback')
    vote_data.nominal('education-spending')
    vote_data.nominal('superfund-right-to-sue')
    vote_data.nominal('crime')
    vote_data.nominal('duty-free-exports')
    vote_data.nominal('export-administration-act-south-africa')

    return vote_data


def load_abalone():
    abalone_file = 'data/raw/abalone.data'
    abalone_data = Data('abalone', abalone_file, 'Rings', names=['Sex', 'Length', 'Diameter',
                                                                 'Height', 'Whole weight',
                                                                 'Shucked weight',
                                                                 'Viscera weight', 'Shell weight',
                                                                 'Rings'])
    abalone_data.nominal('Sex')
    abalone_data.normalization(['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                                'Viscera weight', 'Shell weight', 'Rings'])
    return abalone_data


def load_forest_fire():
    fire_file = 'data/raw/forestfires.data'
    fire_data = Data('forest', fire_file, 'area', transform={'area': np.log},
                     usecols=['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain',
                              'area'])
    fire_data.nominal('month')
    fire_data.nominal('day')
    fire_data.normalization(['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area'])
    return fire_data


def load_machine():
    machine_file = 'data/raw/machine.data'
    machine_data = Data('machine', machine_file, 'PRP', names=['vendor name', 'model', 'MYCT',
                                                               'MMIN', 'MMAX', 'CACH', 'CHMIN',
                                                               'CHMAX', 'PRP', 'ERP'],
                        usecols=['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP'])

    machine_data.normalization(['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP'])

    return machine_data


def split_train_test(train, test, label):
    train_x = train.drop(label, axis=1).to_numpy()
    train_y = train[label].to_numpy()
    test_x = test.drop(label, axis=1).to_numpy()
    test_y = test[label].to_numpy()

    return train_x, train_y, test_x, test_y
