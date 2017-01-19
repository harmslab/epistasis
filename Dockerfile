FROM andrewosh/binder-python-3.5

MAINTAINER Zach Sailer <zsailer@uoregon.edu>

USER root

# Add dependency
RUN apt-get update

USER main

# Install requirements for Python 3
RUN mkdir .github
RUN pip install cython
