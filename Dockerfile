FROM rpy2/jupyter

USER root

RUN apt-get update && apt-get install -y python3-tk build-essential libglu1 gdb

RUN pip3 install --upgrade pip \
    && pip3 --no-cache-dir install scipy scikit-learn matplotlib joblib==0.8.4 seaborn pandas tables statsmodels numba nilearn line_profiler Cython

RUN mkdir /meld && mkdir /meld_work && mkdir /meld_work/work && mkdir /meld_work/work/code && mkdir /meld_work/work/data
COPY . /meld

RUN pip3 --no-cache-dir install /meld

RUN /usr/local/bin/ipython -c "import meld"

RUN chmod -R 777 /meld

ENV JOBLIB_START_METHOD='forkserver' \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ENV PATH=$PATH:/opt/workbench/bin_linux64
USER $NB_USER

ENTRYPOINT ["/usr/local/bin/tini", "--"]
CMD ["start-notebook.sh"]
