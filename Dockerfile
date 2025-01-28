FROM --platform=linux/amd64 nvidia/cuda:12.4.1-base-ubuntu22.04

RUN mkdir /extra
COPY extra /extra/

RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-dev build-essential tcsh perl gzip gcc dc bc libgomp1

RUN pip install --upgrade pip
COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
RUN rm ./requirements.txt

RUN mkdir /INPUTS
RUN mkdir /OUTPUTS

RUN mkdir /project
COPY project /project/

RUN mkdir /scripts
COPY scripts /scripts/

# Only use CPU for now
ENV CUDA_VISIBLE_DEVICES=""

# TODO: Need to decide whether to run inference or training at some point
ENTRYPOINT ["/scripts/pipeline.sh"]
