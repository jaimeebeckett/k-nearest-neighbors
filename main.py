from dataset import Data, ClassificationData
from learning import KNearestNeighbor, CondensedNearestNeighbor, EditedNearestNeighbor
from evaluation import cross_validation, tuning, classification, mse
import numpy as np
import time


def load_car():
    car_file = 'data/raw/car.data'
    car_data = ClassificationData('car', car_file, 'class', names=['buying', 'maint', 'doors',
                                                              'persons', 'lug_boot', 'safety',
                                                              'class'])
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
    cancer_data = ClassificationData('cancer', cancer_file, 'class',
                                names=['Sample Code Number', 'Clump Thickness',
                                       'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                                       'Marginal Adhesion', 'Single Epithelial Cell Size',
                                       'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
                                       'Mitoses', 'class'],
                                usecols=['Clump Thickness', 'Uniformity of Cell Size',
                                         'Uniformity of Cell Shape', 'Marginal Adhesion',
                                         'Single Epithelial Cell Size', 'Bare Nuclei',
                                         'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'class'],
                                na_values='?')
    cancer_data.imputation()
    return cancer_data


def load_vote():
    vote_file = 'data/raw/house-votes-84.data'
    vote_data = ClassificationData('vote', vote_file, 'class',
                              names=['class', 'handicapped-infants', 'water-project-cost-sharing',
                                     'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                                     'el-salvador-aid', 'religious-groups-in-schools',
                                     'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                                     'mx-missile', 'immigration', 'synfuels-corporation-cutback',
                                     'education-spending', 'superfund-right-to-sue', 'crime',
                                     'duty-free-exports', 'export-administration-act-south-africa'],
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
    abalone_data = Data('abalone', abalone_file, 'Rings', names=['Sex', 'Length', 'Diameter', 'Height',
                                                            'Whole weight', 'Shuck ed weight',
                                                            'Viscera weight', 'Shell weight',
                                                            'Rings'])
    abalone_data.nominal('Sex')

    return abalone_data


def load_forest_fire():
    fire_file = 'data/raw/forestfires.data'
    fire_data = Data('forest', fire_file, 'area', transform={'area': np.log},
                     usecols=['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain',
                              'area'])
    return fire_data


def load_machine():
    machine_file = 'data/raw/machine.data'
    machine_data = Data('machine', machine_file, 'PRP', names=['vendor name', 'model', 'MYCT', 'MMIN',
                                                          'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP',
                                                          'ERP'],
                   usecols=['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP'])

    machine_data.normalization(['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP'])
    #machine_data.discretization('MYCT', 5, width=False)

    return machine_data


if __name__ == '__main__':
    car = load_car()
    cancer = load_cancer()
    vote = load_vote()

    abalone = load_abalone()
    forest = load_forest_fire()
    machine = load_machine()

    for data in [car, cancer, vote]:
        knn = KNearestNeighbor(data.label, k=1)
        best_score, best_k = tuning(data, knn, classification)
        print(f'Name: {data.name}')
        print(f'Best Fold: {best_k}')

        scores = cross_validation(knn, data.df, data.folds, data.label, classification)
        print(f'Best Fold: {np.argmax(scores)}, Best Score: {max(scores)}')
        print()

    for data in [abalone, forest, machine]:
        knn = KNearestNeighbor(data.label, k=1)
        best_score = np.inf
        best_sigma = None
        best_k = None
        for sigma in [1, 0.8, 0.5, 0.3, 0.15, 0.1, 0.005, 0.001, 0.0008, 0.0005, 0.0003]:
            score, k = tuning(data, knn, mse, sigma)
            if score < best_score:
                best_score = score
                best_sigma = sigma
                best_k = k
        data.sigma = best_sigma
        data.k = best_k
        print(f'Name: {data.name}')
        print(f'Best k-value: {best_k}, Best Sigma: {best_sigma}')

        scores = cross_validation(knn, data.df, data.folds, data.label, mse)
        print(f'Best Fold: {np.argmin(scores)}, Best Score: {min(scores)}')
        print()

#    print(knn_results)
#    print(cnn_results)
#    print(enn_results)
#    print(f"Best knn k-value, epsilon, sigma: {knn_k, knn_epsilon, knn_sigma}")
#    print(f"Best cnn k-value epsilon, sigma: {cnn_k, cnn_epsilon, cnn_sigma}")
#    print(f"Best enn k-value epsilon, sigma: {enn_k, enn_epsilon, enn_sigma}")
#
#     knn_results = {
#         'Fold #': [],
#         'Training Size': [],
#         'Test Size': [],
#         'Accuracy of Error': []
#     }
#     cnn_results = {
#         'Fold #': [],
#         'Training Size': [],
#         'Test Size': [],
#         'Accuracy of Error': []
#     }
#
#     enn_results = {
#         'Fold #': [],
#         'Training Size': [],
#         'Test Size': [],
#         'Accuracy of Error': []
#     }
#     knn_k = 11
#     cnn_k = 9
#     enn_k = 19
#     cnn_train = condensed_edited_nearest_neighbor(tuning_train, label, k=3)
#     enn_train = condensed_edited_nearest_neighbor(tuning_train, label, k=1, condensed=False)
#
#     for i in range(len(folds)):
#         train = data.drop(folds[i].index)
#         test = folds[i]
#         start = timer()
#         knn_predicted = k_nearest_neighbors(train, test, label, classification, knn_k)
#         end = timer()
#         print(f'knn time: {end - start}')
#         start = timer()
#         cnn_predict = k_nearest_neighbors(cnn_train, test, label, classification, cnn_k)
#         end = timer()
#         print(f'cnn time: {end - start}')
#         start = timer()
#         enn_predict = k_nearest_neighbors(enn_train, test, label, classification, enn_k)
#         end = timer()
#         print(f'enn time: {end - start}')
#         test_labels = test[label].to_list()
#         knn_eval_results = eval_metric(test_labels, knn_predicted, eval_type)
#         cnn_eval_results = eval_metric(test_labels, cnn_predict, eval_type)
#         enn_eval_results = eval_metric(test_labels, enn_predict, eval_type)
#
#         knn_results['Fold #'].append(i)
#         knn_results['Training Size'].append(len(train))
#         knn_results['Test Size'].append(len(test))
#         knn_results['Accuracy of Error'].append(knn_eval_results)
#
#         cnn_results['Fold #'].append(i)
#         cnn_results['Training Size'].append(len(train))
#         cnn_results['Test Size'].append(len(test))
#         cnn_results['Accuracy of Error'].append(cnn_eval_results)
#
#         enn_results['Fold #'].append(i)
#         enn_results['Training Size'].append(len(train))
#         enn_results['Test Size'].append(len(test))
#         enn_results['Accuracy of Error'].append(enn_eval_results)
#
#     print()
#     print(f'knn_results: {knn_results}')
#     print(f'cnn_results: {cnn_results}')
#     print(f'enn_results: {enn_results}')
#     print(f"Best knn fold: {max(knn_results, key=knn_results.get)}")
#     print(f"Best cnn fold: {max(cnn_results, key=cnn_results.get)}")
#     print(f"Best enn fold: {max(enn_results, key=enn_results.get)}")
#
#     print()
#     print('knn:')
#     print(f'The Best Eval Score: {max(knn_results["Accuracy of Error"])}')
#     print(
#         f'The Average of All Eval Scores: {sum(knn_results["Accuracy of Error"]) / len(knn_results["Accuracy of Error"])}')
#     print()
#     print('cnn:')
#     print(f'The Best Eval Score: {max(cnn_results["Accuracy of Error"])}')
#     print(
#         f'The Average of All Eval Scores: {sum(cnn_results["Accuracy of Error"]) / len(cnn_results["Accuracy of Error"])}')
#     print()
#     print('enn:')
#     print(f'The Best Eval Score: {max(enn_results["Accuracy of Error"])}')
#     print(
#         f'The Average of All Eval Scores: {sum(enn_results["Accuracy of Error"]) / len(enn_results["Accuracy of Error"])}')
