import google.protobuf
import pandas as pd
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

totalmovie_df=pd.read_csv("total_movie_d8.csv", index_col=0)
starscore_df=pd.read_csv("star_score_d8.csv", index_col=0)

test_totalmovie_df=pd.read_csv("test_total_movie_d8.csv", index_col=0)
test_starscore_df=pd.read_csv("test_star_score_d8.csv", index_col=0)


result_df = pd.merge(totalmovie_df, starscore_df, on='movieCd')
test_result_df = pd.merge(test_totalmovie_df, test_starscore_df, on='movieCd')

result_df.dropna(axis=0)
test_result_df.dropna(axis=0)

result_df["sumd8_audi"]=result_df["preview_audience"]+result_df["d1_audience"]+result_df["d2_audience"]+result_df["d3_audience"]+result_df["d4_audience"]+result_df["d5_audience"]+result_df["d6_audience"]+result_df["d7_audience"]+result_df["d8_audience"]
test_result_df.to_csv("mid_result_d8_df.csv",encoding="utf-8")

result_df.drop(['movieCd', 'd1_audience','d2_audience','d3_audience','d4_audience','d5_audience','d6_audience','d7_audience','d8_audience',
               'd1_screen','d2_screen','d3_screen','d4_screen','d5_screen','d6_screen','d7_screen','d8_screen',
               'd1_show','d2_show','d3_show','d4_show','d5_show','d6_show','d7_show','d8_show',
               'd1_seat','d2_seat','d3_seat','d4_seat','d5_seat','d6_seat','d7_seat','d8_seat','audience', 'movieNm_y'], axis=1, inplace=True)

test_result_df.drop(['movieCd', 'movieNm_y'], axis=1, inplace=True)

result_df.columns=['cat1','cat2','cont1','cont2','cat3','cat4','cont3','cat5','cat6','cat7','cat8','cat9','cont4','cont5','cont6','sumd8_audi']
test_result_df.columns=['cat1','cat2','cont1','cont2','cat3','cat4','cont3','cat5','cat6','cat7','cat8','cat9','cont4','cont5','cont6']


#result_df.replace(np.nan, 'missing')
result_df=result_df.fillna('missing')
test_result_df=result_df.fillna('missing')

result_df.to_csv("result_d8_df.csv",encoding="utf-8")
test_result_df.to_csv("test_result_d8_df.csv",encoding="utf-8")

#df_train_ori = pd.read_csv('result_d8_df.csv', keep_default_na=False)
df_train_ori = pd.read_csv('result_d8_df.csv')
#df_train_ori.replace(np.nan, 'missing')
df_test_ori = pd.read_csv('test_result_d8_df.csv')


train_df = df_train_ori.head(2000)
evaluate_df = df_train_ori.tail(400)
test_df=df_test_ori.head(2)

MODEL_DIR = "tf_model_full"

print("train_df.shape = ", train_df.shape)
print("evaluate_df.shape = ", evaluate_df.shape)
print("test_df.shape = ", test_df.shape)

features = train_df.columns
#categorical_features = features['movieNm_x','director','repNationNm', 'repGenreNm', 'watchGradeNm', 'actor_1', 'actor_2', 'actor_3', 'companyNm']
#categorical_features = features[1]+features[2]
#continuous_features = features['openDt', 'prdtYear', 'showTm', 'preview_audience', 'star_score', 'star_user_count']
#continuous_features = features['openDt', 'prdtYear', 'showTm', 'preview_audience', 'star_score', 'star_user_count']
#print(categorical_features)

#categorical_features = [feature for feature in features if 'movieNm_x'or'director'or'repNationNm'or'repGenreNm'or'watchGradeNm'or'actor_1'or'actor_2'or'actor_3'or'companyNm' in feature]
#categorical_features = [feature for feature in features if 'movieNm_x' in feature]

#categorical_features = ['movieNm_x', 'director', 'repNationNm', 'repGenreNm', 'watchGradeNm', 'actor_1', 'actor_2', 'actor_3', 'companyNm']
categorical_features = [feature for feature in features if 'cat' in feature]
#continuous_features = ['openDt', 'prdtYear', 'showTm', 'preview_audience', 'star_score', 'star_user_count']
continuous_features = [feature for feature in features if 'cont' in feature]
LABEL_COLUMN='sumd8_audi'


# Converting Data into Tensors
def input_fn(df, training=True):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in continuous_features}

    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
        for k in categorical_features}

    # Merges the two dictionaries into one.
    feature_cols = dict(list(continuous_cols.items()) +
                        list(categorical_cols.items()))

    if training:
        # Converts the label column into a constant Tensor.
        label = tf.constant(df[LABEL_COLUMN].values)

        # Returns the feature columns and the label.
        return feature_cols, label

    # Returns the feature columns
    return feature_cols


def train_input_fn():
    return input_fn(train_df)


def eval_input_fn():
    return input_fn(evaluate_df)


def test_input_fn():
    return input_fn(test_df, False)

engineered_features = []

for continuous_feature in continuous_features:
    engineered_features.append(
        tf.contrib.layers.real_valued_column(continuous_feature))


for categorical_feature in categorical_features:
    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        categorical_feature, hash_bucket_size=1000)

    engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16,
                                                                  combiner="sum"))

regressor = tf.contrib.learn.DNNRegressor(
    feature_columns=engineered_features, hidden_units=[10, 10], model_dir=MODEL_DIR)

# Training Our Model
wrap = regressor.fit(input_fn=train_input_fn, steps=500)

# Evaluating Our Model
print('Evaluating ...')
results = regressor.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

predicted_output = regressor.predict(input_fn=test_input_fn)

y=regressor.predict_scores(input_fn=test_input_fn)

print(list(predicted_output))

