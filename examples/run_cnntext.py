from xatc.models.cnntext import train, Predictor

train(data_path="../data/tiny_train_data.txt",
      model_path="../models/cnntext_model.bin",
      train_size=0.9,
      model_params={
          "epochs": 1,
          "batch_size": 64,
          "lr":1e-4,
      })


predictor = Predictor(model_path="../models/cnntext_model.bin")
pred = predictor.predict(data=["我是句子"])

print(pred)
