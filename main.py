from processing import load_data, imputation, ordinal, nominal, standardization, discretization
from learning import k_nearest_neighbors, k_fold, condensed_nearest_neighbor, nearest_neighbor_cross_validation
from evaluation import majority_prediction, eval_metric
import numpy as np

if __name__ == '__main__':
    # classification files

    # car data
    car_file = 'data/raw/car.data'
    car = load_data(car_file, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
    car_column_order = {'buying': {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
                        'maint': {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
                        'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
                        'persons': {'2': 0, '4': 1, 'more': 3},
                        'lug_boot': {'small': 0, 'low': 0, 'med': 1, 'high': 2, 'big': 2},
                        'safety': {'small': 0, 'low': 0, 'med': 1, 'high': 2, 'big': 2}}
    car = ordinal(car, car_column_order)

    # cancer data
    #cancer_file = 'data/raw/breast-cancer-wisconsin.data'
    #cancer = load_data(cancer_file, names=['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size',
    #                                        'Uniformity of Cell Shape', 'Marginal Adhesion',
    #                                        'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
    #                                        'Normal Nucleoli', 'Mitoses', 'class'],
    #                    usecols=['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
    #                             'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
    #                             'Normal Nucleoli', 'Mitoses', 'class'], na_values='?')
    # cancer = imputation(cancer)
    #
    # # vote data
    # vote_file = 'data/raw/house-votes-84.data'
    # vote = load_data(vote_file, names=['class', 'handicapped-infants', 'water-project-cost-sharing',
    #                                    'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
    #                                    'religious-groups-in-schools', 'anti-satellite-test-ban',
    #                                    'aid-to-nicaraguan-contras', 'mx-missile', 'immigration',
    #                                    'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue',
    #                                    'crime', 'duty-free-exports', 'export-administration-act-south-africa'],
    #                  na_values='?')
    # vote = imputation(vote)
    # vote = nominal(vote, [col for col in vote.columns if col != 'class'])
    #
    # # regression files
    # abalone_file = 'data/raw/abalone.data'
    # fire_file = 'data/raw/forestfires.data'
    # machine_file = 'data/raw/machine.data'
    #
    # # load data
    # abalone = load_data(abalone_file, names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
    #                                          'Viscera weight', 'Shell weight', 'Rings'])
    # forest = load_data(fire_file, transform={'area': np.log}, usecols=['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp',
    #                                                                    'RH', 'wind', 'rain', 'area'])
    # machine = load_data(machine_file, names=['vendor name', 'model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX',
    #                                          'PRP', 'ERP'],
    #                     usecols=['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP'])
    # abalone = nominal(abalone, ['Sex'])
    # machine = discretization(machine, 'MYCT', 5, width=False)

    print('Car Dataset:')
    data = car
    label = 'class'
    eval_type = 'classification'
    folds, validation = k_fold(data, 'class', validation=True)
    validation_folds = k_fold(validation, 'class')[0]
    validation_train = data.drop(validation_folds[0].index)
    validation_test = validation_folds[0]
    validation_true_labels = validation_test[label].to_list()
    knn_results = {}
    cnn_results = {}
    #enn_results = {}
    for i in range(1, 10):
        knn_prediction = k_nearest_neighbors(validation_train, validation_test, label, k=i)
        cnn_prediction = condensed_nearest_neighbor(validation_train, validation_test, label, k=i)
        #enn_prediction

        knn_eval = eval_metric(validation_true_labels, knn_prediction, eval_type)
        cnn_eval = eval_metric(validation_true_labels, cnn_prediction, eval_type)

        knn_results[i] = knn_eval
        cnn_results[i] = cnn_eval
        #enn_results[i] = enn_eval

    print(f"Best knn k-value: {max(knn_results, key=knn_results.get)}")
    print(f"Best cnn k-value: {max(cnn_results, key=knn_results.get)}")

   # knn_k, cnn_k = nearest_neighbor_cross_validation(data, validation_folds, 'class')
    print('done')
    results = {
        'Fold #': [],
        'Training Size': [],
        'Test Size': [],
        'Accuracy of Error': []
    }
    for i in range(knn_k):
        train = data.drop(folds[i].index)
        test = folds[i]
        knn_predicted = k_nearest_neighbors(train, test, 'class', knn_k)
        cnn_predicted = condensed_nearest_neighbor(train, test, 'class', cnn_k)
        test_labels = test['class'].to_list()
        knn_eval_results = eval_metric(test_labels, knn_predicted, 'classification')
        cnn_eval_results = eval_metric(test_labels, cnn_predicted, 'classification')

        results['Fold #'].append(i)
        results['Training Size'].append(len(train))
        results['Test Size'].append(len(test))
        results['Accuracy of Error'].append(knn_eval_results)

    print()
    print(f'The Best Eval Score: {max(results["Accuracy of Error"])}')
    print(f'The Average of All Eval Scores: {sum(results["Accuracy of Error"]) / len(results["Accuracy of Error"])}')
