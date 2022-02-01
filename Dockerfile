FROM docker.lco.global/banzai:1.7.3-1-gd14a68a

USER root

RUN conda install -y coveralls sphinx docutils=0.15 cython

COPY --chown=10087:10000 . /lco/banzai-floyds

RUN apt-get -y update && apt-get -y install gcc && \
        pip install --no-cache-dir git+https://github.com/lcogt/banzai.git@feature/no-archive-token /lco/banzai-floyds/ && \
        apt-get -y remove gcc && \
        apt-get autoclean && \
        rm -rf /var/lib/apt/lists/*

RUN chown -R archive /home/archive

USER archive
