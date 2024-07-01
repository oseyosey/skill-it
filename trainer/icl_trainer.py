from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import wandb 
import pandas as pd

from .utils import get_train_dataset, get_tokenized_train_dataset, get_steps, create_optimizer_scheduler, get_train_dataloader, log_val_loss_per_skill
from .trainer import AbstractTrainer 

class ICLTrainer(AbstractTrainer):
    def train(
        self,
        args,
        logger,
        tokenizer,
        model,
        validation_data,
        evaluator, 
    ):
        """Modified Standard Pytorch training and evaluation code to enable in-context learning."""
        tokenized_val, output_idxs = validation_data.get_tokenized_dataset() # only when calling this, will either the validation/train data be tokenized.
        train_data = get_train_dataset(args, logger, tokenizer)
        n_data = args.n_select if args.n_select != 0 else args.max_steps * args.batch_size
        tokenized_train = get_tokenized_train_dataset(args, train_data, n_data)
        
        train_dataloader = get_train_dataloader(args.task_name, tokenizer, tokenized_train, args.batch_size, args.slicer)   
        
        ckpt_steps, total_steps = get_steps(args)
        
        progress_bar = tqdm(range(total_steps))
        logging_steps = 10
        counter = 0

        ## TODO: Create custom ICL dataset based on validation and train data.

        #loss_dict = evaluator.evaluate(tokenized_val_icl, counter, None, output_idxs_icl) 
    
        loss_dict = evaluator.evaluate(tokenized_val, counter, None, output_idxs) 
        log_val_loss_per_skill(logger, loss_dict) 