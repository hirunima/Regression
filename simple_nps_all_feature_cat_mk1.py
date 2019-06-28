from __future__ import print_function
import numpy as np
import os
import sys
import time
import pandas as pd 
import keras
from tqdm._tqdm_notebook import tqdm_notebook
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score, recall_score, f1_score
from keras import callbacks
from sklearn.utils import resample
import logging
import sys
import argparse
np.set_printoptions(threshold=sys.maxsize)


def load_data():
    stime = time.time()
    # print(os.listdir(INPUT_PATH))
    df_ge = pd.read_csv("Virtual_NPS_dataset.csv",encoding='latin1', engine='c')
    print(df_ge.shape)
    print(df_ge.columns)
    print(df_ge.head(5))
    tqdm_notebook.pandas('Processing...')


    train_cols=["short_answer","survey_name","sentiment","call_count_6","duration_queue_sum_6","duration_tot_sum_6","visit_count","totaldurwait_sum_6","totaldurserve_sum_6","break_count","complete_duration_sum_6","arpu","network_stay","credit_category","credit_type","customer_priority_type","age","gender","device_brand","device_type","account_type","lte_flag","ced_selfcare_app_user","ced_online_shopper","ced_overseas_travellers","ced_cricket_lover","ced_social_media","ced_movie_lover","ced_music_lover","cem_news","cem_netflix","ez_customer","star_point_bal","total_idd_rev","total_gprs_volume","voice_rev","sms_rev","nov_d","nov_nd","dec_d","dec_nd","jan_d","jan_nd","feb_d","feb_nd","mar_d","mar_nd","apr_d","apr_nd","nov_od_dur","nov_ond_dur","dec_od_dur","dec_ond_dur","jan_od_dur","jan_ond_dur","feb_od_dur","feb_ond_dur","mar_od_dur","mar_ond_dur","apr_od_dur","apr_ond_dur","nov_id_dur","nov_ind_dur","dec_id_dur","dec_ind_dur","jan_id_dur","jan_ind_dur","feb_id_dur","feb_ind_dur","mar_id_dur","mar_ind_dur","apr_id_dur","apr_ind_dur","nov_od_count","nov_ond_count","dec_od_count","dec_ond_count","jan_od_count","jan_ond_count","feb_od_count","feb_ond_count","mar_od_count","mar_ond_count","apr_od_count","apr_ond_count","nov_id_count","nov_ind_count","dec_id_count","dec_ind_count","jan_id_count","jan_ind_count","feb_id_count","feb_ind_count","mar_id_count","mar_ind_count","apr_id_count","apr_ind_count","lastatlstate","avgatlstate","lastadlstate","avgadlstate"]

    df_data=df_ge.loc[:,train_cols]
    df_data['lastatlstate'] =df_data['lastatlstate'].fillna(2)
    df_data['avgatlstate'] =df_data['avgatlstate'].fillna(2)
    df_data['lastadlstate'] =df_data['lastadlstate'].fillna(2)
    df_data['avgadlstate'] =df_data['avgadlstate'].fillna(2)
    df_data = df_data[df_data['short_answer'].isin(['0','1','2','3','4','5','6','7','8','9','10'])]
    df_data.loc[(df_data.survey_name == 'HOTLINE CSAT') & (df_data.short_answer.astype(int) >= 4), 'sentiment'] = 'POSITIVE'
    df_data.loc[(df_data.survey_name == 'HOTLINE CSAT') & (df_data.short_answer.astype(int) <= 3) & (df_data.short_answer.astype(int) >= 2), 'sentiment'] = 'NATURAL'
    df_data.loc[(df_data.survey_name == 'HOTLINE CSAT') & (df_data.short_answer.astype(int) <= 1), 'sentiment'] = 'NEGATIVE'

    df_data.loc[(df_data.survey_name == 'HOTLINE NPS') & (df_data.short_answer.astype(int) >= 9), 'sentiment'] = 'POSITIVE'
    df_data.loc[(df_data.survey_name == 'HOTLINE NPS') & (df_data.short_answer.astype(int) <= 8) & (df_data.short_answer.astype(int) >= 7), 'sentiment'] = 'NATURAL'
    df_data.loc[(df_data.survey_name == 'HOTLINE NPS') & (df_data.short_answer.astype(int) <= 6), 'sentiment'] = 'NEGATIVE'

    df_data.loc[(df_data.survey_name == 'IVR NPS') & (df_data.short_answer.astype(int) >= 9), 'sentiment'] = 'POSITIVE'
    df_data.loc[(df_data.survey_name == 'IVR NPS') & (df_data.short_answer.astype(int) <= 8) & (df_data.short_answer.astype(int) >= 7), 'sentiment'] = 'NATURAL'
    df_data.loc[(df_data.survey_name == 'IVR NPS') & (df_data.short_answer.astype(int) <= 6), 'sentiment'] = 'NEGATIVE'
    
    df_data.loc[(df_data.ced_selfcare_app_user == 'N'), 'ced_selfcare_app_user'] = 'No'
    df_data.loc[(df_data.ced_online_shopper == 'N'), 'ced_online_shopper'] = 'No'
    
    df_data.loc[(df_data.ced_overseas_travellers == "N"), 'ced_overseas_travellers'] = 'No'
    df_data.loc[(df_data.ced_cricket_lover == 'N'), 'ced_cricket_lover'] = 'No'
    
    df_data.loc[(df_data.ced_social_media == 'N'), 'ced_social_media'] = 'No'
    df_data.loc[(df_data.ced_movie_lover == 'N'), 'ced_movie_lover'] = 'No'
    df_data.loc[(df_data.ced_music_lover == 'N'), 'ced_music_lover'] = 'No'

    df_data=pd.concat([
    df_data.get(["sentiment","call_count_6","duration_queue_sum_6","duration_tot_sum_6","visit_count","totaldurwait_sum_6","totaldurserve_sum_6","break_count","complete_duration_sum_6","network_stay","age","star_point_bal","total_idd_rev","total_gprs_volume","voice_rev","sms_rev","nov_d","nov_nd","dec_d","dec_nd","jan_d","jan_nd","feb_d","feb_nd","mar_d","mar_nd","apr_d","apr_nd","nov_od_dur","nov_ond_dur","dec_od_dur","dec_ond_dur","jan_od_dur","jan_ond_dur","feb_od_dur","feb_ond_dur","mar_od_dur","mar_ond_dur","apr_od_dur","apr_ond_dur","nov_id_dur","nov_ind_dur","dec_id_dur","dec_ind_dur","jan_id_dur","jan_ind_dur","feb_id_dur","feb_ind_dur","mar_id_dur","mar_ind_dur","apr_id_dur","apr_ind_dur","nov_od_count","nov_ond_count","dec_od_count","dec_ond_count","jan_od_count","jan_ond_count","feb_od_count","feb_ond_count","mar_od_count","mar_ond_count","apr_od_count","apr_ond_count","nov_id_count","nov_ind_count","dec_id_count","dec_ind_count","jan_id_count","jan_ind_count","feb_id_count","feb_ind_count","mar_id_count","mar_ind_count","apr_id_count","apr_ind_count","lastatlstate","avgatlstate","lastadlstate","avgadlstate"])
    ,pd.get_dummies(df_data['gender'],prefix='gender')
    # ,pd.get_dummies(df_data['survey_name'],prefix='survey_name')
    ,pd.get_dummies(df_data['credit_category'],prefix='credit_category')
    ,pd.get_dummies(df_data['credit_type'] ,prefix='credit_type')
    ,pd.get_dummies(df_data['customer_priority_type'],prefix='customer_priority_type')
#     ,pd.get_dummies(df_data['device_brand'],prefix='device_brand')
    ,pd.get_dummies(df_data['device_type'],prefix='device_type')
    ,pd.get_dummies(df_data['account_type'],prefix='account_type')
    ,pd.get_dummies(df_data['lte_flag'],prefix='lte_flag')
    ,pd.get_dummies(df_data['ced_selfcare_app_user'],prefix='ced_selfcare_app_user')
    ,pd.get_dummies(df_data['ced_online_shopper'],prefix='ced_online_shopper')
    ,pd.get_dummies(df_data['ced_overseas_travellers'],prefix='ced_overseas_travellers')
    ,pd.get_dummies(df_data['ced_cricket_lover'],prefix='ced_cricket_lover')
    ,pd.get_dummies(df_data['ced_social_media'],prefix='ced_social_media')
    ,pd.get_dummies(df_data['ced_movie_lover'],prefix='ced_movie_lover')
    ,pd.get_dummies(df_data['ced_music_lover'],prefix='ced_music_lover')
    ,pd.get_dummies(df_data['cem_news'],prefix='cem_news')
    ,pd.get_dummies(df_data['cem_netflix'],prefix='cem_netflix')
    ,pd.get_dummies(df_data['ez_customer'],prefix='ez_customer')
    ,pd.get_dummies(df_data["lastatlstate"],prefix='lastatlstate')
    ,pd.get_dummies(df_data["avgatlstate"],prefix='avgatlstate')
    ,pd.get_dummies(df_data["lastadlstate"],prefix='lastadlstate')
    ,pd.get_dummies(df_data["avgadlstate"],prefix='avgadlstate')
    ,pd.get_dummies(df_data["sentiment"],prefix='label')
    ],axis=1)
    le = LabelEncoder()
    df_data['sentiment'] =le.fit_transform(df_data['sentiment'].astype(str))

    print(df_data.dtypes)
    df_data=df_data.apply(pd.to_numeric,errors='coerce')
    print(df_data.head(5))
    df_data.dropna(inplace=True)
    df_train, df_test = train_test_split(df_data, train_size=0.8, test_size=0.2, shuffle=False)
    
    class_count=list(df_train.sentiment.value_counts().values)
    class_values=list(df_train.sentiment.value_counts().index.values)

    df_majority=df_train[df_train.sentiment==class_values[0]]
    df_minority=df_train[df_train.sentiment==class_values[2]]
    df_middle=df_train[df_train.sentiment==class_values[1]]
    # Upsample minority class
    df_minority_upsampled=resample(df_minority, replace=True, n_samples=class_count[1],  random_state=123)
    df_majority_downsampled=resample(df_majority, replace=True, n_samples=class_count[1],  random_state=123)
    # Combine majority class with upsampled minority class
    df_resampled=pd.concat([df_middle,df_majority_downsampled,df_minority_upsampled])
    # Display new class counts
    print(df_resampled.sentiment.value_counts())
    print(df_resampled.dtypes)
    df_resampled.drop('sentiment',axis=1,inplace=True)
    df_test.drop('sentiment',axis=1,inplace=True)
    print("Train--Test size", df_resampled.shape, df_test.shape)
    label_col=[col for col in df_data.columns if 'label' in col]

    # scale the feature MinMax, build array
    y_train = (df_resampled.loc[:,label_col].values).reshape(-1,len(label_col))
    df_resampled.drop(label_col,axis=1,inplace=True)
    print(np.asarray(df_resampled.columns))
    x_train = (df_resampled.values).reshape(-1,df_resampled.shape[1])
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)

    y_test = (df_test.loc[:,label_col].values).reshape(-1,len(label_col))
    df_test.drop(label_col,axis=1,inplace=True)
    x_test = df_test.values.reshape(-1,df_resampled.shape[1])
    x_test = sc.transform(x_test)
    print("Deleting unused dataframes of total size(KB)",(sys.getsizeof(df_data)+sys.getsizeof(df_train)+sys.getsizeof(df_test))//1024)
    print(y_train[:50,:],y_train[:50,:])
    del df_data
    del df_test
    del df_train
    del df_resampled

    print("Batch trimmed size",x_train.shape, y_train.shape)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return (x_train, y_train), (x_test, y_test),label_col

(x_train, y_train), (x_test, y_test),label_col=load_data()

def prediction_model(x_train_shape,len_label_col):

    model = Sequential()
    model.add(Dense(1024, input_dim= x_train_shape,
                            kernel_initializer='random_uniform'))
#     model.add(Dropout(0.4))
    model.add(Dense(512,activation='relu'))
    model.add(BatchNormalization())
#     model.add(Dropout(0.4))
#     model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(len_label_col,activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def train(model,x_train,y_train,x_test,y_test,args):
    log = callbacks.CSVLogger(args.OUTPUT_PATH + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.OUTPUT_PATH + '/weights-{epoch:02d}.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    history=model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              verbose=1,
              validation_data=(x_test, y_test),shuffle=True,callbacks=[log,checkpoint])
    return model,history

def testing(model,x_test, y_test,args,history):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Visualize the training data
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()
    plt.savefig(os.path.join(args.OUTPUT_PATH, 'train_vis_BS_.png'))

    y_pred = model.predict(x_test)
    # y_pred = (y_pred.argmax(axis=1))
    print(y_pred[:10].argmax(axis=1),y_test[:10].argmax(axis=1))
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)
    accuracy=accuracy_score(y_test.argmax(axis=1),y_pred.argmax(axis=1))
    print('Accuracy:%f' %accuracy)
    precision=precision_score(y_test.argmax(axis=1),y_pred.argmax(axis=1),average='micro')
    print('Precision:%f' %precision)
    recall=recall_score(y_test.argmax(axis=1),y_pred.argmax(axis=1),average='micro')
    print('Recall:%f' %recall)
    f1=f1_score(y_test.argmax(axis=1),y_pred.argmax(axis=1),average='micro')
    print('F1score: %f' %f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="virtual nps network")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--OUTPUT_PATH', default='./outputs/model')
    parser.add_argument('--INPUT_PATH', default='./inputs')
    parser.add_argument('--lr', default=0.00001, type=float)
    args = parser.parse_args()
    
    if not os.path.exists(args.OUTPUT_PATH):
        os.makedirs(args.OUTPUT_PATH)
    (x_train, y_train), (x_test, y_test),label_col=load_data()
    
    model = prediction_model(x_train.shape[1],len(label_col))
    model.summary()
    _,history=train(model,x_train,y_train,x_test,y_test,args)
    testing(model,x_test, y_test,args,history)