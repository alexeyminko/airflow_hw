import logging
import os
import json
import dill
import pandas as pd
import datetime as dt

# Укажем путь к файлам проекта:
# -> $PROJECT_PATH при запуске в Airflow
# -> иначе - текущая директория при локальном запуске
path = os.environ.get('PROJECT_PATH', '.')
# path = os.path.expanduser('~/airflow_hw')


def download_model():
    model_lst = os.listdir(f'{path}/data/models')
    model_lst.sort()
    model_name = model_lst[-1]
    model_path = f'{path}/data/models/{model_name}'
    with open(model_path, 'rb') as file:
        model = dill.load(file)
    return model


def get_pred(model, test_path, json_name):
    json_path = test_path + '/' + json_name
    with open(json_path) as fin:
        form = json.load(fin)
        df = pd.DataFrame.from_dict([form])
        y = model.predict(df)
        my_predict = [form['id'], y[0]]
    return my_predict


def download_predictions(results):
    preds_df = pd.DataFrame(results, columns=['car_id', 'pred'])
    preds_df.to_csv(os.path.join(
        f'{path}/data/predictions',
        f'preds_{dt.datetime.now().strftime("%Y%m%d%H%M")}.csv'
    ), index=False)


def predict():

    logging.info('-----Start predict-----')

    model = download_model()

    logging.info('-----Download model completed-----')

    test_path = f'{path}/data/test'
    results = []

    for json_name in os.listdir(test_path):
        results.append(get_pred(model, test_path, json_name))

    logging.info(f'-----Got prediction results: {results}-----')

    download_predictions(results)

    logging.info('-----Download predictions completed-----')


if __name__ == '__main__':
    predict()
