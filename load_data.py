
import pandas as pd

TAG_DICT = {'PER' : '인물', 'ORG' : '단체', 'DAT' : '날짜', 'LOC' : '장소', 'NOH' : '수량' , 'POH' : '기타'}
SUB_TOKEN1 = '→'
SUB_TOKEN2 = '☺'
OBJ_TOKEN1 = '§'
OBJ_TOKEN2 = '↘'

def preprocessing_dataset(dataset):
  subject_entity = []
  object_entity = []
  sen_data = []

  for s, i, j in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
    sub_info=eval(i)
    obj_info=eval(j)

    subject_entity.append(sub_info['word'])
    object_entity.append(obj_info['word'])
        
    sub_type = sub_info['type']
    sub_start = sub_info['start_idx']
    sub_end = sub_info['end_idx']

    obj_type = obj_info['type']
    obj_start = obj_info['start_idx']
    obj_end = obj_info['end_idx']

    sen = add_sep_tok(s, sub_start, sub_end, sub_type, obj_start, obj_end, obj_type)
    sen_data.append(sen)	
    
  out_dataset = pd.DataFrame({'id':dataset['id'],
		'sentence':sen_data,
		'subject_entity':subject_entity,
		'object_entity':object_entity,
		'label':dataset['label'],}
	)
  return out_dataset
  
def load_data(dataset_dir):
	pd_dataset = pd.read_csv(dataset_dir)
	dataset = preprocessing_dataset(pd_dataset)
	return dataset

def add_sep_tok(sen, sub_start, sub_end, sub_type, obj_start, obj_end, obj_type) :
	sub_mid = TAG_DICT[sub_type]
	obj_mid = TAG_DICT[obj_type]

	sub_start_tok = ' ' + SUB_TOKEN1 + ' ' + SUB_TOKEN2 + ' ' + sub_mid + ' ' + SUB_TOKEN2 + ' '
	sub_end_tok = ' ' + SUB_TOKEN1 + ' '
	obj_start_tok = ' ' + OBJ_TOKEN1 + ' ' + OBJ_TOKEN2 + ' ' + obj_mid + ' ' + OBJ_TOKEN2 + ' '
	obj_end_tok = ' ' + OBJ_TOKEN1 + ' '

	if sub_start < obj_start :
		sen = sen[:sub_start] +  sub_start_tok + sen[sub_start:sub_end+1] + sub_end_tok + sen[sub_end+1:]
		obj_start += 13
		obj_end += 13
		sen = sen[:obj_start] + obj_start_tok + sen[obj_start:obj_end+1] + obj_end_tok + sen[obj_end+1:]
	else :
		sen = sen[:obj_start] + obj_start_tok + sen[obj_start:obj_end+1] + obj_end_tok + sen[obj_end+1:]
		sub_start += 13
		sub_end += 13
		sen = sen[:sub_start] + sub_start_tok + sen[sub_start:sub_end+1] + sub_end_tok + sen[sub_end+1:]
	return sen

def tokenized_dataset(dataset, tokenizer, entity_len , max_len, preprocessor):
    entity_data = []
    sen_data = []
    for e01, e02, sen in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
        entity = e01 + ' [SEP] ' + e02
        entity = tokenizer.tokenize(preprocessor(entity))
        entity_data.append(entity)

        sen = preprocessor(sen)
        sen_data.append(sen)

    entity_str_data = []
    for entity in entity_data :
        entity_padded = entity + ['[PAD]'] * (entity_len - len(entity)) if len(entity) <= entity_len \
            else entity[:entity_len]
        entity_str = tokenizer.convert_tokens_to_string(entity_padded)
        entity_str_data.append(entity_str)

    tokenized_sentences = tokenizer(
        entity_str_data, # entity data (subject, object)
        sen_data, # sentence data
		return_tensors="pt",
		truncation=True,
		padding='max_length',
		max_length=max_len,
        return_token_type_ids=False,
		add_special_tokens=True
	)
    return tokenized_sentences


