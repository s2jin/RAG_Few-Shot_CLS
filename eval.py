import os
import json
import sys
import sklearn.metrics as met

def postprocessing(pred):
    result = list()
    for p in pred:
#         elif p in ['독일','중국']:
#             p = '세계'
#         elif p in ['IT','기술']:
#         if p in ['IT','과학']:
        if p in ['IT']:
            p = 'IT과학'
#         elif p in ['문화','생활']:
        elif p in ['생활']:
            p = '생활문화'
#         elif p in ['금융', '생활경제']:
#             p = '경제'
#         elif p in ['사회과학','사회공헌']:
#             p = '사회'
        result.append(p)
    return result

if __name__=='__main__':
    
    filename = sys.argv[1]
    ext = os.path.splitext(filename)[-1]
    if ext == '.jsonl':
        with open(filename, 'r') as f:
            data = [json.loads(d) for d in f]
        pred = [d['output']['preds'] for d in data if 'output' in d]
        gold = [d['output']['labels'] for d in data if 'output' in d]
        srce = [d['output']['inputs'] for d in data if 'output' in d]

    pred = postprocessing(pred)

    assert len(pred) == len(gold) and len(gold) == len(srce)
    print('data size: {}\n'.format(len(gold)))
    label_list = sorted(set(gold))
    print('gold:',label_list)
    print('pred:',sorted(set(pred)))

    print(met.classification_report(gold, pred, digits=4, labels=label_list))
