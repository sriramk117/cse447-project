FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
RUN pip install tqdm