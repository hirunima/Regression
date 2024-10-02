import numpy as np
import os
import sys
import time
import pandas as pd 
from tqdm._tqdm_notebook import tqdm_notebook
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout,BatchNormalization
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import logging
import sys
np.set_printoptions(threshold=sys.maxsize)
# import talos as ta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

params = {
    "batch_size": 50,  
    "epochs": 25,
    "lr": 0.0001,
#     "time_steps": 60
}

#iter_changes = "dropout_layers_0.4_0.4"
PATH_TO_DRIVE_ML_DATA='.'
INPUT_PATH = PATH_TO_DRIVE_ML_DATA+"/inputs"
OUTPUT_PATH = PATH_TO_DRIVE_ML_DATA+"/outputs/model1"#+iter_changes
# TIME_STEPS = params["time_steps"]
BATCH_SIZE = params["batch_size"]
stime = time.time()

# check if directory already exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print("Directory created", OUTPUT_PATH)

stime = time.time()
# print(os.listdir(INPUT_PATH))
df_ge = pd.read_csv("dataset.csv",encoding='latin1', engine='c')
print(df_ge.shape)
print(df_ge.columns)
print(df_ge.head(5))
tqdm_notebook.pandas('Processing...')

train_cols=["survey_name","sentiment","call_count_6","duration_queue_sum_6","duration_tot_sum_6","visit_count","totaldurwait_sum_6","totaldurserve_sum_6","break_count","complete_duration_sum_6","arpu","network_stay","credit_category","credit_type","customer_priority_type","age","gender","device_brand","device_type","account_type","lte_flag","ced_selfcare_app_user","ced_online_shopper","ced_overseas_travellers","ced_cricket_lover","ced_social_media","ced_movie_lover","ced_music_lover","cem_news","cem_netflix","ez_customer","star_point_bal","total_idd_rev","total_gprs_volume","voice_rev","sms_rev","nov_d","nov_nd","dec_d","dec_nd","jan_d","jan_nd","feb_d","feb_nd","mar_d","mar_nd","apr_d","apr_nd","nov_od_dur","nov_ond_dur","dec_od_dur","dec_ond_dur","jan_od_dur","jan_ond_dur","feb_od_dur","feb_ond_dur","mar_od_dur","mar_ond_dur","apr_od_dur","apr_ond_dur","nov_id_dur","nov_ind_dur","dec_id_dur","dec_ind_dur","jan_id_dur","jan_ind_dur","feb_id_dur","feb_ind_dur","mar_id_dur","mar_ind_dur","apr_id_dur","apr_ind_dur","nov_od_count","nov_ond_count","dec_od_count","dec_ond_count","jan_od_count","jan_ond_count","feb_od_count","feb_ond_count","mar_od_count","mar_ond_count","apr_od_count","apr_ond_count","nov_id_count","nov_ind_count","dec_id_count","dec_ind_count","jan_id_count","jan_ind_count","feb_id_count","feb_ind_count","mar_id_count","mar_ind_count","apr_id_count","apr_ind_count","lastatlstate","avgatlstate","lastadlstate","avgadlstate"]

df_data=df_ge.loc[:,train_cols]
df_data['lastatlstate'] =df_data['lastatlstate'].fillna(2)
df_data['avgatlstate'] =df_data['avgatlstate'].fillna(2)
df_data['lastadlstate'] =df_data['lastadlstate'].fillna(2)
df_data['avgadlstate'] =df_data['avgadlstate'].fillna(2)

df_data=pd.concat([
df_data.get(["call_count_6","duration_queue_sum_6","duration_tot_sum_6","visit_count","totaldurwait_sum_6","totaldurserve_sum_6","break_count","complete_duration_sum_6","network_stay","age","star_point_bal","total_idd_rev","total_gprs_volume","voice_rev","sms_rev","nov_d","nov_nd","dec_d","dec_nd","jan_d","jan_nd","feb_d","feb_nd","mar_d","mar_nd","apr_d","apr_nd","nov_od_dur","nov_ond_dur","dec_od_dur","dec_ond_dur","jan_od_dur","jan_ond_dur","feb_od_dur","feb_ond_dur","mar_od_dur","mar_ond_dur","apr_od_dur","apr_ond_dur","nov_id_dur","nov_ind_dur","dec_id_dur","dec_ind_dur","jan_id_dur","jan_ind_dur","feb_id_dur","feb_ind_dur","mar_id_dur","mar_ind_dur","apr_id_dur","apr_ind_dur","nov_od_count","nov_ond_count","dec_od_count","dec_ond_count","jan_od_count","jan_ond_count","feb_od_count","feb_ond_count","mar_od_count","mar_ond_count","apr_od_count","apr_ond_count","nov_id_count","nov_ind_count","dec_id_count","dec_ind_count","jan_id_count","jan_ind_count","feb_id_count","feb_ind_count","mar_id_count","mar_ind_count","apr_id_count","apr_ind_count","lastatlstate","avgatlstate","lastadlstate","avgadlstate"])
,pd.get_dummies(df_data['gender'],prefix='gender')
,pd.get_dummies(df_data['survey_name'],prefix='survey_name')
,pd.get_dummies(df_data['credit_category'],prefix='credit_category')
,pd.get_dummies(df_data['credit_type'] ,prefix='credit_type')
,pd.get_dummies(df_data['customer_priority_type'],prefix='customer_priority_type')
,pd.get_dummies(df_data['device_brand'],prefix='device_brand')
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

print(df_data.dtypes)
df_data=df_data.apply(pd.to_numeric,errors='coerce')
df_data.dropna(inplace=True)
df_train, df_test = train_test_split(df_data, train_size=0.8, test_size=0.2, shuffle=False)
print("Train--Test size", df_train.shape, df_test.shape)
label_col=[col for col in df_data.columns if 'label' in col]

# scale the feature MinMax, build array
y_train = (df_train.loc[:,label_col].values).reshape(-1,len(label_col))
df_train.drop(label_col,axis=1,inplace=True)
x_train = (df_train.values).reshape(-1,df_train.shape[1])
min_max_scaler = MinMaxScaler()

x_train = min_max_scaler.fit_transform(x_train)

y_test = (df_test.loc[:,label_col].values).reshape(-1,len(label_col))
df_test.drop(label_col,axis=1,inplace=True)
x_test = df_test.values.reshape(-1,df_train.shape[1])
x_test = min_max_scaler.transform(x_test)
print("Deleting unused dataframes of total size(KB)",(sys.getsizeof(df_data)+sys.getsizeof(df_train)+sys.getsizeof(df_test))//1024)
print(y_train[:10,:])
del df_data
del df_test
del df_train

print("Batch trimmed size",x_train.shape, y_train.shape)

def create_model():
    model = Sequential()
    # (batch_size, timesteps, data_dim)
    model.add(Dense(1024, input_dim= x_train.shape[1],
                        kernel_initializer='random_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.4))
#     model.add(Dense(512,activation='relu'))
#     model.add(Dropout(0.4))
    model.add(Dense(128,activation='relu'))
#     model.add(BatchNormalization(0.8))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(len(label_col),activation='softmax'))
    optimizer = optimizers.adam(lr=params["lr"])
    model.compile(loss='categorical_crossentropy',optimizer=optimizer)
    return model

model=None
try:
    model = pickle.load(open("lstm_model", 'rb'))
    print("Loaded saved model...")
except FileNotFoundError:
    print("Model not found")


print("Test size", x_test.shape, y_test.shape)
    
is_update_model = True
if model is None or is_update_model:
    from keras import backend as K
    print("Building model...")
    print("checking if GPU available", K.tensorflow_backend._get_available_gpus())
    model = create_model()
    model.summary()
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=40, min_delta=0.0001)
    
    mcp = ModelCheckpoint(os.path.join(OUTPUT_PATH,
                          "best_model.h5"), monitor='val_loss', verbose=1,
                          save_best_only=True, save_weights_only=False, mode='min', period=1)

    # Not used here. But leaving it here as a reminder for future
    r_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, 
                                  verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    
    csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'training_log_' + time.ctime().replace(" ","_") + '.log'), append=True)

       # Training without data augmentation:
    history=model.fit(x_train, y_train, epochs=params["epochs"], verbose=2, batch_size=BATCH_SIZE,
                        shuffle=False, validation_data=(x_test,y_test))
#     callbacks=[log,checkpoint, lr_decay],
    
    print("saving model...")
    pickle.dump(model, open("model", "wb"))

# model.evaluate(x_test_t, y_test_t, batch_size=BATCH_SIZE
y_pred = model.predict(x_test, batch_size=BATCH_SIZE)
#y_pred = y_pred.flatte/
#y_test = trim_dataset(y_test, BATCH_SIZE)
# error = mean_squared_error(y_test[0:len(y_pred)], y_pred)
# print("Error is", error, y_pred.shape, y_test.shape)
print(y_pred[0:15])
print(y_test[0:15])

# convert the predicted value to range of real data
y_pred_org = (y_pred * min_max_scaler.data_range_[0]) + min_max_scaler.data_min_[0]
# min_max_scaler.inverse_transform(y_pred)
y_test_t_org = (y_test * min_max_scaler.data_range_[0]) + min_max_scaler.data_min_[0]
# min_max_scaler.inverse_transform(y_test_t)
print(y_pred_org[0:15])
print(y_test_t_org[0:15])

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
plt.savefig(os.path.join(OUTPUT_PATH, 'train_vis_BS_'+str(BATCH_SIZE)+"_"+time.ctime()+'.png'))

# load the saved best model from above
saved_model = load_model(os.path.join(OUTPUT_PATH, 'best_model.h5')) # , "lstm_best_7-3-19_12AM",
print(saved_model)

y_pred = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
y_pred = y_pred.flatten()
y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
error = mean_squared_error(y_test_t, y_pred)
print("Error is", error, y_pred.shape, y_test_t.shape)
print(y_pred[0:15])
print(y_test_t[0:15])
y_pred_org = (y_pred * min_max_scaler.data_range_[0]) + min_max_scaler.data_min_[0] # min_max_scaler.inverse_transform(y_pred)
y_test_t_org = (y_test_t * min_max_scaler.data_range_[0]) + min_max_scaler.data_min_[0] # min_max_scaler.inverse_transform(y_test_t)
print(y_pred_org[0:15])
print(y_test_t_org[0:15])

y_pred = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
y_pred = y_pred.flatten()
y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
y_test_t_org = (y_test_t * min_max_scaler.data_range_[0]) + min_max_scaler.data_min_[0] # min_max_scaler.inverse_transform(y_test_t)

# Visualize the prediction
from matplotlib import pyplot as plt
plt.figure()
plt.plot(y_pred_org)
plt.plot(y_test_t_org)
plt.title('Prediction vs Real Call Volume')
plt.ylabel('Call Volume')
plt.xlabel('time')
plt.legend(['Prediction', 'Real'], loc='upper left')
#plt.show()
plt.savefig(os.path.join(OUTPUT_PATH, 'pred_vs_real_BS'+str(BATCH_SIZE)+"_"+time.ctime()+'.png'))

plt.figure()
plt.plot(x_train)
plt.plot(y_test_t_org)
plt.title('Training vs Real Call Volume')
plt.ylabel('Call Volume')
plt.xlabel('time')
plt.legend(['Train', 'Real'], loc='upper left')
#plt.show()
plt.savefig(os.path.join(OUTPUT_PATH, 'train_vs_real_BS'+str(BATCH_SIZE)+"_"+time.ctime()+'.png'))
print_time("program completed ", stime)
