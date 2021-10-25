import pandas as pd
import os
from sampling import sampler
from cleansing import tab_clean
from Image_data_handle import image_sampler
from feat_eng import feat_engine
from model import Model
from pred import predictor
import warnings

warnings.filterwarnings("ignore")
cwd = os.getcwd()


def main():
    path = r'D:\Hackthons\Shell_Cloud\Shell_Cloud\dataset'
    wd = os.path.join(cwd, path)

    train_image_dir = os.path.join(wd,'train')
    train_tabular = pd.read_csv(wd + r'\train\train.csv')

    # function to clean Total Cloud Cover outliers
    train_tabular = tab_clean(train_tabular)
    test_scenarios = os.listdir(wd + r'\test')

    submission = list()

    for num,scenario in enumerate(test_scenarios):
        print('Predicting for Test Scenario:', num)

        test_tabular = pd.read_csv(os.path.join(os.path.join(wd,'test'), scenario) + r'\weather_data.csv')

        # iterate over train one 1yr data and subset data samples based on cloud cover correlation
        train_df_list = sampler(train_tabular, test_tabular)

        # get image data for the train subset -- may ignore for now
        # train_image_df_list = image_sampler(train_image_dir, train_df_list)

        # feat engineering -- create new features if any
        X_train, Y_train = feat_engine(train_df_list)

        # first prepare input train data samples based on train_df_list and train model
        # TODO
        # https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
        # Multiple Input Multi-Step Output
        model = Model(X_train, Y_train)

        # predictions -- write out submissions
        predictions = predictor(model, test_tabular)

        submission.append(predictions)
        break

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
