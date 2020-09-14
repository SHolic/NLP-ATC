from xatc.models.fasttext import train, predict


train(data_path="../data/addr_train_data.txt",
      model_path="../models/fasttext_model.bin",
      train_size=0.9,
      model_params=None)

pred = predict(data=["上海嘉宁国际大厦30楼"],
               model_path="../models/fasttext_model.bin",
               return_type=None)
print(pred)

# pred = predict(data=["我想写书出版", "上周去日本玩了"],
#                model_path="../models/fasttext_model.bin",
#                return_type="sent_embedding")
# print(pred)
#
# pred = predict(data=["出版", "上周"],
#                model_path="../models/fasttext_model.bin",
#                return_type="word_embedding")
# print(pred)


""" result
[FasttextDatasets.load]: run time is 155.328 s
[FastText.train]: run time is 141.06 s
[FastText.predict]: run time is 21.42 s
[FastText.predict]: run time is 2.587 s
[Fasttext evaluation] Train data accuracy is: 0.975069
[Fasttext evaluation] Test data accuracy is:  0.791216
[FastText.save]: run time is 1.56 s
[FastText.load]: run time is 0.909 s
[FastText.predict]: run time is 0.0 s
['公司地址']
"""