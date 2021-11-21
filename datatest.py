
import re
import unittest
import pandas as pd
from transformers import AutoTokenizer
from load_data import load_data, tokenized_dataset
from preprocessor import Preprocessor
from dataset import RE_Dataset
from train import label_to_num


class DataTest(unittest.TestCase) :

    def setUp(self) :
        self.data = load_data("/opt/ml/project/RelationExtraction/data/train/train.csv")
        self.label = label_to_num(self.data['label'].values)
        
        self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
        self.preprocessor = Preprocessor(self.tokenizer)

        tokenized_data = tokenized_dataset(dataset=self.data, 
            tokenizer=self.tokenizer, 
            entity_len=10,
            max_len=256, 
            preprocessor=self.preprocessor
        )
        
        # -- Dataset
        self.dataset = RE_Dataset(tokenized_data, self.label)

    def test_type(self) :
        self.assertIsInstance(self.data , pd.DataFrame)
        self.assertIsInstance(self.label , list)

    def test_columens(self) :
        column_list = list(self.data.columns)
        self.assertTrue('id' in column_list)
        self.assertTrue('sentence' in column_list)
        self.assertTrue('subject_entity' in column_list)
        self.assertTrue('object_entity' in column_list)
        self.assertTrue('label' in column_list)

    def test_label_range(self) :
        for num in self.label :
            self.assertTrue(0 <= num < 30)

    def test_preprocessor(self) :
        sen_list = list(self.data['sentence'])
        unk_chars = re.compile('[\u3000-\u303f\ud800—\udbff\ue000—\uf8ff]')
        outrange_chars = re.compile('[\uffff-\U000e007f]')

        for sen in sen_list :
            sen = self.preprocessor(sen)
            self.assertTrue(unk_chars.match(sen) == None)
            self.assertTrue(outrange_chars.match(sen) == None)

    def test_sep(self) :
        dataset = self.dataset.pair_dataset
        for input_id in dataset['input_ids'] :
            self.assertEqual(input_id[10+1].item(), self.tokenizer.sep_token_id)


if __name__ == '__main__' :  
    unittest.main()