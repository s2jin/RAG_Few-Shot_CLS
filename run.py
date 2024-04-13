import os
import sys
import hydra

import random
import numpy as np
import torch
from omegaconf import DictConfig, open_dict

import logging
logging.basicConfig(level=logging.INFO)

# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir

@hydra.main(version_base='1.2', config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    logging.info('COMMAND: python '+' '.join(sys.argv))
    if config.seed != None:
        logging.info(f'SET seed {config.seed}')
        set_seed(config.seed)

    if config.job_num == None:
        config.job_num = ''
    else:
        config.job_num = str(config.job_num)

    logging.info('<CONFIG>\n'+'\n'.join(print_config(config)))

    with open_dict(config):
        config.hydra = hydra.core.hydra_config.HydraConfig.get()

    agent = hydra.utils.get_class(config.agent._target_)
    agent = agent(**config)
    agent.run()

    return None 

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.set_num_threads(1)
    #torch.set_num_interop_threads(1)

def print_config(config_dict, level=0):
    if type(config_dict) != dict:
        config_dict = dict(config_dict)
    result = list()
    for key in config_dict:
        if type(config_dict[key]) == DictConfig:
            result.append(f"{'    '*level}[ {key} ]:\t(dict)")
            result += print_config(config_dict[key], level=level+1)
        else:
            result.append(f"{'    '*level}[ {key} ]:\t({type(config_dict[key]).__name__})\t{config_dict[key]}")
    return result

if __name__ == "__main__":
    main()
