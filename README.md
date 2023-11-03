# sentiment-clf
Text sentiment classification – [wandb report](https://api.wandb.ai/links/hzchet/4y33f0rx).

# Setup
First you should clone this repository by running
```shell
git clone https://github.com/hzchet/sentiment-clf.git
cd sentiment-clf
```
Then modify Makefile – change DATA_DIR (where your data is located) and SAVE_DIR (where you want to save all the logs and checkpoints) and also specify the index of your local GPU.

# Training
In order to train models with your own configurations, run the following command

```bash
python3 train.py --config src/configs/my_config.json
```
`src/configs/my_config.json` – is where you should specify the model architecture and training details.
