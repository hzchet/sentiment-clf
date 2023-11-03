import argparse
import json
import os
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.data_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer
    tokenizer = config.get_tokenizer()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, tokenizer)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, num_embeddings=len(tokenizer))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for key in ['yelp']:
            logger.info(f'{key}:')
            results = []
            for batch_num, batch in enumerate(tqdm(dataloaders[key])):
                batch = Trainer.move_batch_to_device(batch, device)
                output = model(**batch)
                if type(output) is dict:
                    batch.update(output)
                else:
                    batch["logits"] = output

                batch["argmax"] = batch["logits"].argmax(-1)
                for i in range(len(batch["text"])):
                    argmax = int(batch["argmax"][i].detach().cpu().numpy())
                    results.append(argmax)
            
            with Path(f'{key}_{out_file}').open("w") as f:
                for pred in results:
                    f.write(str(pred) + '\n')


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.txt",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    main(config, args.output)
