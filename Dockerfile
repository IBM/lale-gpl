ARG python_version=3.7.3
FROM python:${python_version}-stretch

RUN echo 'deb http://lib.stat.cmu.edu/R/CRAN/bin/linux/debian stretch-cran35/' >> /etc/apt/sources.list

RUN apt-get update \
&& apt-get install --assume-yes openjdk-8-jdk graphviz build-essential swig \
&& apt-get install --assume-yes dirmngr apt-transport-https ca-certificates software-properties-common gnupg2 \
&& apt-get --assume-yes --allow-unauthenticated install r-base -t stretch-cran35

RUN Rscript -e 'install.packages("arules", repos="http://cran.r-project.org")'
RUN Rscript -e 'install.packages("arulesCBA", repos="https://cran.rstudio.com")'

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

RUN pip install numpy
COPY setup.py README.md /lalegpl/
WORKDIR /lalegpl
# First install the dependencies
RUN pip install .[test]

COPY . /lalegpl

RUN pip install .[test]

ENV PYTHONPATH /lalegpl

