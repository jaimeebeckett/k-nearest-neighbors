from dataset import Data, ClassificationData
from learning import NearestNeighbor, CondensedNearestNeighbor, EditedNearestNeighbor
from evaluation import cross_validation, classification, mse
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
                                                            'Whole weight', 'Shucked weight',
                                                            'Viscera weight', 'Shell weight',
                                                            'Rings'])
    abalone_data.nominal('Sex')
    abalone_data.normalization(['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                                'Viscera weight', 'Shell weight', 'Rings'])
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

    def split_train_test(dataset):
        label = dataset.label
        dataset.k_fold(validation=True)

        train_x = dataset.validation_train.drop(label, axis=1).to_numpy()
        train_y = dataset.validation_train[label].to_numpy()
        test_x = dataset.validation_test.drop(label, axis=1).to_numpy()
        test_y = dataset.validation_test[label].to_numpy()

        return train_x, train_y, test_x, test_y

    def evaluate(nn, train_x, train_y, test_x, test_y, evaluation_method):
        prediction = nn.predict(train_x, train_y, test_x)
        return evaluation_method(test_y, prediction)

    def regression_tune(nn, train_x, train_y, test_x, test_y, evaluation_method):
        nn_score = np.inf
        nn_sigma = None
        for s in [1, 0.8, 0.5, 0.3, 0.15, 0.1, 0.005, 0.001, 0.0008, 0.0005, 0.0003]:
            nn.sigma = s
            eval_score = evaluate(nn, train_x, train_y, test_x, test_y, evaluation_method)
            if eval_score < nn_score:
                nn_score = eval_score
                nn_sigma = s

        return nn_score, nn_sigma


    for data in [car, cancer, vote, abalone, forest, machine]:
        class_data = isinstance(data, ClassificationData)
        evaluation = classification if class_data else mse
        best_k_knn = 0
        best_sigma_knn = None
        best_k_cnn = 0
        best_sigma_cnn = None
        best_k_enn = 0
        best_sigma_enn = None
        best_score_knn = 0 if class_data else np.inf
        best_score_cnn = 0 if class_data else np.inf
        best_score_enn = 0 if class_data else np.inf
        train, train_label, test, test_label = split_train_test(data)
        knn = NearestNeighbor(data.label, k=1)
        cnn = CondensedNearestNeighbor(data.label, k=1)
        # enn = CondensedNearestNeighbor(data.label, k=1)

        for i in range(1, 15, 2):
            knn.k = i
            cnn.k = i
            if class_data:
                knn_score = evaluate(knn, train, train_label, test, test_label, evaluation)
                cnn_score = evaluate(cnn, train, train_label, test, test_label, evaluation)
                if knn_score > best_score_knn:
                    best_score_knn = knn_score
                    best_k_knn = i
                if cnn_score > best_score_cnn:
                    best_score_cnn = cnn_score
                    best_k_cnn = i
            else:
                knn_score, knn_sigma = regression_tune(knn, train, train_label, test, test_label, evaluation)
                cnn_score, cnn_sigma = regression_tune(cnn, train, train_label, test, test_label, evaluation)
                if knn_score < best_score_knn:
                    best_score_knn = knn_score
                    best_k_knn = i
                    best_sigma_knn = knn_sigma
                if cnn_score < best_score_cnn:
                    best_score_cnn = cnn_score
                    best_k_cnn = i
                    best_sigma_cnn = cnn_sigma
            # if (class_data and score > best_score_knn) or (not class_data and score < best_score_knn):
            #     best_score_knn = score
            #     best_k_knn = i
            #     if not class_data:
            #         best_sigma_knn = sigma

        knn.k = best_k_knn
        knn.sigma = best_sigma_knn
        cnn.k = best_k_cnn
        cnn.sigma = best_sigma_cnn
        # enn.k = beset_k_enn

        knn_scores = knn.cross_validation(data.df, data.folds, data.label, evaluation)
        cnn_scores = cnn.cross_validation(data.df, data.folds, data.label, evaluation)
        # enn_scores = enn.cross_validation(data.df, data.folds, data.label, evaluation)

        print()
        print(f'Name: {data.name}')
        print('K-Nearest Neighbors:')
        print(f'Best K: {best_k_knn}')
        print(f'Best Fold: {np.argmax(knn_scores)}, Best Score: {max(knn_scores)}')
        print()
        print('Condensed Nearest Neighbors:')
        print(f'Best K: {best_k_cnn}')
        print(f'Best Fold: {np.argmax(cnn_scores)}, Best Score: {max(cnn_scores)}')
        print()
        print('Edited Nearest Neighbors:')
        print(f'Best K: {best_k_enn}')
       # print(f'Best Fold: {np.argmax(enn_scores)}, Best Score: {max(enn_scores)}')
        print()


    # for data in [abalone, forest, machine]:
    #     knn = NearestNeighbor(data.label, k=1)
    #     best_score = np.inf
    #     best_sigma = None
    #     best_k = None
    #     for sigma in [1, 0.8, 0.5, 0.3, 0.15, 0.1, 0.005, 0.001, 0.0008, 0.0005, 0.0003]:f
    #         knn.sigma = sigma
    #         score, k = tuning(data, knn, mse, sigma=sigma)
    #         if score < best_score:
    #             best_score = score
    #             best_sigma = sigma
    #             best_k = k
    #     data.sigma = best_sigma
    #     data.k = best_k
    #     print(f'Name: {data.name}')
    #     print(f'Best k-value: {best_k}, Best Sigma: {best_sigma}')
    #
    #     scores = cross_validation(knn, data.df, data.folds, data.label, mse)
    #     print(f'Best Fold: {np.argmin(scores)}, Best Score: {min(scores)}')
    #     print()
