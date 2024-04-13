import transformers
from omegaconf import DictConfig

import logging
logging.basicConfig(level=logging.INFO)

class Tokenizer():
    def __init__(self, 
            path,
            **kwargs,
            ):
        self.config = DictConfig(kwargs)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        logging.info(f'LOAD tokenizer: {path}')

    def get_tokenizer(self):
        symbols = '!#$%&\"\'()*+,-./0123456789;:<=>?@'
        hangle = '다람쥐 헌 쳇바퀴에 타고파'
        alphabet  = 'The Quick Brown Fox Jumps Over The Lazy Dog'
        for sample in [symbols, hangle, alphabet]:
            logging.info(f"\nTokenized sample:\n{sample}\n=> {self.tokenizer.tokenize(sample)}")
        return self.tokenizer
        
