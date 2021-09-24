from processing import load_data, imputation, ordinal, nominal, standardization, discretization
from learning import k_nearest_neighbors, k_fold
from evaluation import majority_prediction, eval_metric
import numpy as np

if __name__ == '__main__':
    # classification files
    car_file = 'data/raw/car.data'
    cancer_file = 'data/raw/breast-cancer-wisconsin.data'
    vote_file = 'data/raw/house-votes-84.data'
    # regression files
    abalone_file = 'data/raw/abalone.data'
    fire_file = 'data/raw/forestfires.data'
    machine_file = 'data/raw/machine.data'

    car = load_data(car_file, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
    cancer = load_data(cancer_file, names=['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size',
                                           'Uniformity of Cell Shape', 'Marginal Adhesion',
                                           'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                                           'Normal Nucleoli', 'Mitoses', 'class'],
                       usecols=['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                                'Normal Nucleoli', 'Mitoses', 'class'], na_values='?')
    vote = load_data(vote_file, names=['class', 'handicapped-infants', 'water-project-cost-sharing',
                                       'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
                                       'religious-groups-in-schools', 'anti-satellite-test-ban',
                                       'aid-to-nicaraguan-contras', 'mx-missile', 'immigration',
                                       'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue',
                                       'crime', 'duty-free-exports', 'export-administration-act-south-africa'],
                     na_values='?')
    abalone = load_data(abalone_file, names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                                             'Viscera weight', 'Shell weight', 'Rings'])
    forest = load_data(fire_file, transform={'area': np.log}, usecols=['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp',
                                                                       'RH', 'wind', 'rain', 'area'])
    machine = load_data(machine_file, names=['vendor name', 'model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX',
                                             'PRP', 'ERP'],
                        usecols=['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP'])

    # handle missing values
    cancer = imputation(cancer)
    vote = imputation(vote)

    # handle categorical data
    # ordinal
    car_column_order = {'buying': {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
                        'maint': {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
                        'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
                        'persons': {'2': 0, '4': 1, 'more': 3},
                        'lug_boot': {'small': 0, 'low': 0, 'med': 1, 'high': 2, 'big': 2},
                        'safety': {'small': 0, 'low': 0, 'med': 1, 'high': 2, 'big': 2}}
    car = ordinal(car, car_column_order)
    # nominal
    vote = nominal(vote, [col for col in vote.columns if col != 'class'])
    abalone = nominal(abalone, ['Sex'])

    # discretization
    machine = discretization(machine, 'MYCT', 5, width=False)

    print('Car Dataset:')
    data = car

    best_k = 1
    best_result = 0
    folds, validation = k_fold(data, 'class', validation=True)
    if not validation.empty:
        validation_folds = k_fold(validation, 'class')[0]
        k_values = [1, 2, 3, 4, 5]
        for i in range(len(k_values)):
            x = validation_folds[i]
            train = data.drop(validation_folds[i].index)
            test = validation_folds[i]
            predicted = k_nearest_neighbors(train, test, 'class', k_values[i])
            test_labels = test['class'].to_list()
            results = eval_metric(test['class'].to_list(), predicted, 'classification')
            if results > best_result:
                best_k = k_values[i]
                best_result = results

    print(f"Best k value: {best_k}")
    results = {
        'Fold #': [],
        'Training Size': [],
        'Test Size': [],
        'Accuracy of Error': []
    }
    for i in range(5):
        train = data.drop(folds[i].index)
        test = folds[i]
        predicted = k_nearest_neighbors(train, test, 'class', best_k)
        test_labels = test['class'].to_list()
        eval_results = eval_metric(test_labels, predicted, 'classification')
        results['Fold #'].append(i)
        results['Training Size'].append(len(train))
        results['Test Size'].append(len(test))
        results['Accuracy of Error'].append(eval_results)

    print()
    print(f'The Best Eval Score: {max(results["Accuracy of Error"])}')
    print(f'The Average of All Eval Scores: {sum(results["Accuracy of Error"])/len(results["Accuracy of Error"])}')

