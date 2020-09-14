from xatc.models.bert_classifier import train, Predictor


train(data_path="../data/tiny_train_data.txt",
      model_path="../models/bert_model.bin",
      train_size=0.9,
      model_params={
          "epochs": 2,
          "batch_size": 128,
          "hidden_dim": 50,
      })

predictor = Predictor(model_path="../models/bert_model.bin")
pred = predictor.predict(data=["上海嘉宁国际大厦30楼"])
#
# # pred = predict(data=["上海嘉宁国际大厦30楼"], model_path="../models/bert_model.bin")
# print(pred)
#
# pred = predict(data_path="../data/tiny_train_data.txt", model_path="../models/bert_model.bin")
# print(pred)
