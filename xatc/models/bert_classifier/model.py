import torch
from transformers import BertModel

from pathlib import PurePath

CURR_PATH = PurePath(__file__).parent
PRE_TRAINED_MODEL_PATH = str(CURR_PATH.parent.parent / "res/bert_base_wwm/")


class BertClassificationModel(torch.nn.Module):
    def __init__(self,
                 num_labels,
                 finetune=True,
                 label_weights=None,
                 hidden_dim=50,
                 dropout=0.3,
                 pre_trained_model_path=PRE_TRAINED_MODEL_PATH):
        super(BertClassificationModel, self).__init__()
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(pre_trained_model_path)
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden = torch.nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, self.num_labels)
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = torch.nn.MSELoss()

        self.finetune = finetune
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, word_seq_tensor, word_mask_tensor, labels=None):
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(input_ids=word_seq_tensor, attention_mask=word_mask_tensor)
        else:
            outputs = self.bert(input_ids=word_seq_tensor, attention_mask=word_mask_tensor)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        pooled_output = torch.nn.functional.relu(self.hidden(pooled_output))
        pooled_output = self.dropout(pooled_output)
        logits = self.softmax(self.classifier(pooled_output))

        outputs = (logits,)

        if labels is not None:
            loss = self.loss(logits, labels)
            outputs = outputs + (loss,)
        else:
            outputs = outputs + (None,)
        return outputs  # logits, loss
