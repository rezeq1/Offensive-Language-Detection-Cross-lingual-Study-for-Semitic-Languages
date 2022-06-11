from simpletransformers.classification import ClassificationModel
import os
import math
import datetime
import multiprocessing
from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from multiprocessing import Process, freeze_support

x = {"reprocess_input_data": True,
     "fp16": False,
     "num_train_epochs": 8,
     "use_cuda": False}

model = ClassificationModel(
    "bert", "outputs/",
    num_labels=2,
    args=x,
    use_cuda=False
)


def predict(model, text):
    res = model.predict([text])
    return res
    # ------------- End Mine -------------------------




if __name__ == '__main__':
    print(predict(model, "בבחירות הבאות שיילכו לעזאזל הם ובנט כולם הבייתה שקרנים עלובים"))
    # print(predict('כדור בראש וסלמאת'))
    # app.run(debug=True)
    print("Load bert finished ...")

