FROM docker.lco.global/banzai:1.7.2

USER root

RUN conda install -y coveralls sphinx docutils=0.15

COPY --chown=10087:10000 . /lco/banzai-floyds

RUN pip install /lco/banzai-floyds/ --no-cache-dir

RUN chown -R archive /home/archive

USER archive
