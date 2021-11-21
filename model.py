import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from transformers import RobertaModel, RobertaPreTrainedModel

@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class LSTMHead(nn.Module) :
    def __init__(self, layer_size, feature_size, intermediate_size) :
        super(LSTMHead, self).__init__()
        self.layer_size = layer_size
        self.feature_size = feature_size
        self.intermediate_size = intermediate_size

        self.lstm = torch.nn.LSTM(input_size=feature_size,
            hidden_size = intermediate_size,
            num_layers=layer_size,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        self.init_weights()

    def init_weights(self) :
        for p in self.parameters() :
            if p.requires_grad == True and p.dim() > 1:
                nn.init.kaiming_uniform_(p)

    def forward(self, x) :
        batch_size = x.shape[0] 

        h_input = torch.zeros((2*self.layer_size, batch_size, self.intermediate_size))
        c_input = torch.zeros((2*self.layer_size, batch_size, self.intermediate_size))

        if torch.cuda.is_available() :
            h_input = h_input.cuda()
            c_input = c_input.cuda()

        y, (h_output, c_output) = self.lstm(x, (h_input,c_input))
        return y

class LSTMForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, model_name, config):
        super(LSTMForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = RobertaModel.from_pretrained(model_name, 
            config=config, 
            add_pooling_layer=False
        )

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.lstm_head = LSTMHead(layer_size=config.head_layer_size,
            feature_size=config.hidden_size,
            intermediate_size=config.head_hidden_size,
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.head_hidden_size*2, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[0][:,0].unsqueeze(1) # encoded vector of cls token (batch_size, 1, hidden_size)

        pooled_output = self.dropout(pooled_output) # dropout layer
        pooled_output = self.lstm_head(pooled_output).squeeze(1) # lstm head (batch_size, feature_size)
        logits = self.classifier(pooled_output) # (batch_size, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
