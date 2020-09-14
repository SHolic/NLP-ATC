import torch
import time


class CNNText(torch.nn.Module):
    def __init__(self,
                 num_labels,
                 finetune=True,
                 label_weights=None,
                 hidden_dim=50,
                 dropout=0.3,
                 embeddings=None,
                 kernel_size=None,
                 kernel_num=100):
        super(CNNText, self).__init__()
        if kernel_size is None:
            kernel_size = [3, 4, 5]
        self.kernel_size = kernel_size
        self.num_labels = num_labels
        self.finetune = finetune
        self.hidden_dim = hidden_dim
        self.label_weights = label_weights
        # self.vocab_path = vocab_path
        # self.pre_trained_embedding_path = pre_trained_embedding_path
        #
        # if pre_trained_embedding_path is not None:
        #     self.emb_loader = EmbeddingDatasets(embedding_path=pre_trained_embedding_path, sep=" ")
        # else:
        #     self.emb_loader = EmbeddingDatasets(vocab_path=vocab_path, embedding_dim=embedding_dim)
        # print(torch.tensor(self.emb_loader.embeddings))
        self.embed = torch.nn.Embedding.from_pretrained(torch.tensor(embeddings))
        self.emb_dim = torch.tensor(embeddings).shape[1]

        self.dropout = torch.nn.Dropout(dropout)
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(1, kernel_num, (ks, self.emb_dim)) \
                                          for ks in kernel_size])
        self.classifier = torch.nn.Linear(len(kernel_size) * kernel_num, self.num_labels)
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = torch.nn.MSELoss()
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, sent_ids, labels=None):
        x = self.embed(sent_ids)

        if self.finetune is False:
            x = torch.autograd.Variable(x, requires_grad=False)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        con_x = [conv(x) for conv in self.convs]
        pool_x = [torch.nn.functional.max_pool1d(c.squeeze(-1), c.size()[2]) for c in con_x]
        linear_pool = torch.cat(pool_x, 1)

        linear_pool = self.dropout(linear_pool)
        linear_pool = linear_pool.squeeze(-1)

        logits = self.softmax(self.classifier(linear_pool))  # (N, C)

        outputs = (logits,)

        if labels is not None:
            loss = self.loss(logits, labels)
            outputs = outputs + (loss,)
        else:
            outputs = outputs + (None,)
        return outputs  # logits, loss
