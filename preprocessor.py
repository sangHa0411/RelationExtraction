import re

class Preprocessor :
    def __init__(self, tokenizer) :
        self.kor_processor = re.compile(self.buile_list(tokenizer))
    
    def buile_list(self, tokenizer) :
        kor_chars = []
        start_idx = ord('가')
        end_idx = ord('힣')

        for i in range(start_idx, end_idx+1) :
            ch = chr(i)
            if tokenizer.convert_tokens_to_ids(ch) == 1 :
                kor_chars.append(ch)

        kor_ch_list = '[' + ''.join(kor_chars) + ']'
        return kor_ch_list

    def add_space(self, match) :
        sep_ch = match.group()
        return ' ' + sep_ch + ' '

    def __call__(self, sen) :
        assert isinstance(sen, str)
        sen = re.sub('[À-ͳẠ-ỹΑ-ΰか-ヿ]' , '' , sen)
        sen = self.kor_processor.sub(self.add_space, sen)
        return sen
