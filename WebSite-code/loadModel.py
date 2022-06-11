from tensorflow import keras
from keras.models import model_from_json
import joblib
import SVM_model as svm_model
import load_bert_model as bert_model
# def load_svm_model(filepath):
#     # Load SVM model
#     decisiontree_model = open(filepath, "rb")
#     model = joblib.load(decisiontree_model)
#     return model


# def load_lstm_model():
#     # load json and create model
#     print("Loaded model from disk")
#     json_file = open('static\models\lstm_model.json',
#                      'r')  # Change the name of the file
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     # Change the name of the file
#     loaded_model.load_weights("static\models\lstm_model.h5")
#     print("Loaded model from disk")
#     return loaded_model


# '''
# # Test -  loading svm model and try to predict 
# print("--------------------------  Start Loading svm model  --------------------------   ")

# # Load svm model
# model = loadmodel("C:\\Users\\user\\Desktop\\FPFE\\static\\models\\svm_model.pkl")
# # tokenizer 

# # preprocessing 

# # predict - model.predict 

# print("--------------------------  End Loading svm model  --------------------------   ")
# # End

# '''
# lstm = load_lstm_model()
# print(lstm.summary())

def load_model(model_name , text):
    if model_name == 'SVM For Arabic':
        return svm_model.predict(text)
    if model_name == 'BERT For Hebrew':
        bert_model.predict(text)
    if model_name == 'SVM for Hebrew':
        # Todo add the svm for hebrew 
        pass
    