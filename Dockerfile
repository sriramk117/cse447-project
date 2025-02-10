FROM python:3.10-slim
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
RUN pip install tqdm
RUN pip install torch
