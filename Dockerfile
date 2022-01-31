FROM docker.lco.global/banzai:1.7.2

USER root

RUN conda install -y coveralls sphinx docutils=0.15

RUN apt-get -y update && apt-get -y install gcc && \
        pip install --no-cache-dir git+https://github.com/lcogt/banzai.git && \
        apt-get -y remove gcc && \
        apt-get autoclean && \
        rm -rf /var/lib/apt/lists/*

COPY --chown=10087:10000 . /lco/banzai-floyds

RUN pip install /lco/banzai-floyds/ --no-cache-dir

RUN chown -R archive /home/archive

USER archive
