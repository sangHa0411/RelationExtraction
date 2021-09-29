import re

def preprocess_sen(sen) :
    sen = re.sub('[ぁ-ㄬ㐀-鿕]+', ' [CHN] ' , sen)
    sen = re.sub('[^가-힣0-9a-zA-Z\[\]\',.!?]' , ' ', sen)
    sen = re.sub(' {2,}' , ' ' , sen)
    sen = sen.strip()
    return sen