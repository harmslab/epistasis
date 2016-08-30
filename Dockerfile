FROM andrewosh/binder-base

MAINTAINER Zach Sailer <zsailer@uoregon.edu>

USER root

# Add dependency
RUN apt-get update

USER main

# Install requirements for Python 2 and 3
RUN mkdir .github
RUN git clone https://github.com/harmslab/seqspace .github/seqspace
RUN pip install -e .github/seqspace
#RUN /home/main/anaconda/envs/python3/bin/pip install -e .github/seqspace
RUN pip install -e .
#RUN /home/main/anaconda/envs/python3/bin/pip install -e .
