def eval_metric(true, predicted, method):
    if method == "classification":
        return sum(t == p for t, p in zip(true, predicted)) / len(true)
    elif method == "mse":
        return (sum((t - p)**2 for t, p in zip(true, predicted))/ len(true))
    else:
        print(f'{method} is not a valid method')
        return


def majority_prediction(df_train, df_test, label, classification=True):
    if classification:
        value = df_train[label].value_counts().idxmax()
    else:
        value = df_train[label].mode()[0]
    return [value] * len(df_train)
