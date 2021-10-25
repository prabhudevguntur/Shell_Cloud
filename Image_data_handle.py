import os
import pandas as pd
import numpy as np
import cv2

def image_sampler(train_image_dir, train_df_list):

    df_list = []
    for df in train_df_list:
        row_list = []
        for i,j in zip(df['DATE (MM/DD)'],df['MST']):
            # print(i,j)

            if int(i.split('/')[0]) < 10:
                a = '0'+ i.split('/')[0]
            else:
                a
            if int(i.split('/')[1]) < 10:
                b = '0' + i.split('/')[1]
            else:
                b

            date = a+b
            time = j.replace(':','')
            nearest_time = find_nearest_integer_time(time)

            date_time_jpg = date + str(nearest_time) + '.jpg'
            img = cv2.imread(os.path.join(os.path.join(train_image_dir,date),date_time_jpg), cv2.IMREAD_GRAYSCALE)
            img_readj = np.sum(img,axis=0)
            row_list.append(img_readj)
        try:
            img_df = pd.DataFrame(row_list)
            df_list.append(img_df)
        except:
            df_list.append(np.nan)
    return df_list

def find_nearest_integer_time(time):
    int_time = int(time)
    rounded = round(int_time / 10) * 10
    rounded = rounded * 100
    return rounded