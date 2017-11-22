import google.protobuf
import pandas as pd
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

totalmovie_df=pd.read_csv("total_movie_d14.csv", index_col=0)
starscore_df=pd.read_csv("star_score_d14.csv", index_col=0)

result_df = pd.merge(totalmovie_df, starscore_df, on='movieCd')
result_df.dropna(axis=0)
result_df["sumd14_audi"]=result_df["preview_audience"]+result_df["d1_audience"]+result_df["d2_audience"]+result_df["d3_audience"]+result_df["d4_audience"]+result_df["d5_audience"]+result_df["d6_audience"]+result_df["d7_audience"]+result_df["d8_audience"]+result_df["d9_audience"]+result_df["d10_audience"]+result_df["d11_audience"]+result_df["d12_audience"]+result_df["d13_audience"]+result_df["d14_audience"]

result_df.to_csv("result_df_d14.csv",encoding="utf-8")