FROM registry.cerebra.kz/ml/cerebra:latest

RUN apt update && apt install -y sudo

ARG UNAME=testuser
ARG UID=1000
ARG GID=1000

ENV WANDB_API_KEY=?

RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME && echo "$UNAME:123" | chpasswd && adduser $UNAME sudo
USER $UNAME

RUN pip install addict
RUN pip install wandb
