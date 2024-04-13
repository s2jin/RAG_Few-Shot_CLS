import torch
import transformers
import numpy as np
from omegaconf import DictConfig

import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm

class BertRetrieval(transformers.BertPreTrainedModel):
    
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.config_retriever = DictConfig(kwargs['retriever'])

        self.tokenizer = kwargs.pop('tokenizer', None)

        self.block_records = None 
        self.register_buffer( ## layer처럼 동작하지만 optimizer에 의해 업데이트되지는 않음
                "block_embeds",
                torch.zeros(()).new_empty(
                    size=(self.config_retriever.num_kbs, self.config_retriever.proj_size),
                    dtype=torch.float32,
                ),
            )

        self.bert = transformers.BertModel(self.config) ## Retriever Embedder
        if self.config.hidden_size != self.config_retriever.proj_size:
            self.projector = torch.nn.Linear(self.config.hidden_size, self.config_retriever.proj_size)
        else:
            self.projector = None

        self.post_init() # Initialize weights and apply final processing

    def set_kb_embeddings(self,
            block_records,
            **kwargs):

        if self.tokenizer == None:
            raise AttributeError(f'"set_external_embeddings()" requires tokenizer. But no tokenizer loaded in M3BertRetriever')

        self.block_records = block_records
        documents = [d.decode() for d in block_records]
        batch_size = kwargs.pop('batch_size', 128)

        def chunks(data, n):
            for i in range(0, len(data), n): yield data[i:i+n]

        for bindex, batch in tqdm(enumerate(chunks(documents, batch_size)), desc='SET external knowledge'):
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = inputs.to(self.bert.device)
            with torch.no_grad():
                embeds = self.get_query_embedding(**inputs)
                self.block_embeds[bindex*batch_size: bindex*batch_size+batch_size] = embeds

        logging.info(f'SET external knowledge: {self.block_embeds.shape}')

    def get_relevance_score(self,
            query_embeddings=None,
            kb_embeddings=None,
            **kwargs):

#         qeury_embeddings = torch.nn.functional.normalize(query_embeddings, dim=1)
#         kb_embeddings = torch.nn.functional.normalize(kb_embeddings, dim=1)

        relevance_score = torch.einsum("BD,QD->QB", kb_embeddings, query_embeddings) ## (num_kb, d_model) x (batch, d_model) = (batch, num_kb)
        relevance_score = torch.softmax(relevance_score, dim=1)
        topk_relevance_score, topk_relevance_index = torch.topk(relevance_score, k=self.config_retriever.num_candidates, dim=1)  ## (batch, topk)

        return {'score':topk_relevance_score, 'index':topk_relevance_index}
        

    def get_query_embedding(self,
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
            **kwargs):

        embedder_outputs = self.bert(
                input_ids, ## tokenized input text
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )
        pooled_output = embedder_outputs[1]
        if self.projector is not None:
            logits = self.projector(pooled_output)
        else:
            logits = pooled_output

        return logits

    def get_concat_inputs(self,
            input_ids,
            retrieved_ids,
            return_str=None,
            return_ids=None,
            **kwargs):

        if self.tokenizer == None:
            raise AttributeError(f"Tokenizer was not given when Retriever was created. \"get_concat_inputs()\" is required Tokenizer.")

        input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        retrieved_ids = retrieved_ids.tolist()
        retrieved_text = np.take(self.block_records, indices=retrieved_ids, axis=0)
        
        text = list()
        text_pair = list()
        for i in range(retrieved_text.shape[1]):
            for j in range(retrieved_text.shape[0]):
                text.append(input_text[j])
                text_pair.append(retrieved_text[j][i].decode())
        return {'inputs':text, 'kbs':text_pair}

    def get_merge_inputs(self,
            input_ids=None,
            retrieved_ids=None,
            return_str=None,
            return_ids=None,
            **kwargs):

        if self.tokenizer == None:
            raise AttributeError(f"Tokenizer was not given when Retriever was created. \"get_concat_inputs()\" is required Tokenizer.")

        input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        retrieved_ids = retrieved_ids.tolist()
        retrieved_text = np.take(self.block_records, indices=retrieved_ids, axis=0)
        
        text = list()
        text_pair = list()

        for i in range(retrieved_text.shape[0]):
            text.append(input_text[i])
            #text.append(input_text[i].replace('< s > ','<s>').replace('< / s > ','</s>').replace(' < t >', ' <t>'))
            retrieved_doc = [retrieved_text[i][j].decode() for j in range(retrieved_text.shape[1])]
            text_pair.append(retrieved_doc)
        return {'inputs':text, 'kbs':text_pair}

                
        
    def forward(self,
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
            return_str=False,
            return_id=None,
            retriever_concat=False,
            input_text=None,
            **kwargs):

        query_embed = self.get_query_embedding( input_ids = input_ids )

        relevance_outputs = self.get_relevance_score(query_embed, self.block_embeds)
        relevance_score = relevance_outputs['score']
        relevance_index = relevance_outputs['index']

        '''
        ## relevance_score랑 relevance_logits가 동일한 값이 나오기 때문에 아래 코드 사용 X

        block_embeds = self.block_embeds.unsqueeze(0).repeat_interleave(input_ids.size(0), dim=0) ## (num_kb, d_model) -> (batch, num_kb, d_model)
        #expanded_relevance_index = relevance_index.unsqueeze(-1).expand(-1,-1,block_embeds.size(-1)) ## (batch, topk) -> (batch, topk, d_model)
        retrieved_doc_embeds = block_embeds.gather(1, relevance_index.unsqueeze(-1).expand(-1,-1,block_embeds.size(-1))) ## (batch, topk, d_model)
        relevance_logits = torch.einsum("BDC,BCK->BDK", retrieved_doc_embeds, query_embed.unsqueeze(2)) ## (batch, topk, d_model) x (batch, d_model, 1) = (batch, topk, 1)
        '''
        
        if retriever_concat:
            concat_inputs = self.get_concat_inputs( input_ids, relevance_index, 
                    return_str=return_str, 
                    return_id=return_id
                    )
        else:
            concat_inputs = self.get_merge_inputs( 
                    input_ids=input_ids, 
                    input_text=input_text,
                    retrieved_ids=relevance_index, 
                    return_str=return_str, 
                    return_id=return_id
                    )
            

        return RetrieverOutput(
                relevance_score=relevance_score,
                relevance_index=relevance_index,
                concat_inputs=concat_inputs,
                retrieved_doc=None,
                )


class RetrieverOutput(transformers.utils.ModelOutput):
    relevance_score = None
    relevance_index = None
    concat_inputs = None
    retrieved_doc = None
