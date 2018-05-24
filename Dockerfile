FROM rpy2/jupyter

USER root

RUN apt-get update && apt-get install -y python3-tk build-essential libglu1 gdb

RUN pip3 install --upgrade pip \
    && pip3 --no-cache-dir install scipy scikit-learn matplotlib joblib==0.8.4 seaborn pandas tables statsmodels numba nilearn line_profiler Cython

RUN mkdir /meld
COPY . /meld

RUN pip3 --no-cache-dir install /meld

RUN /usr/local/bin/ipython -c "import meld" && mkdir -p /meld_root/work/data /meld_root/work/code

RUN chmod -R 777 /meld && chmod -R 777 /meld_root 

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ENV PATH=$PATH:/opt/workbench/bin_linux64
USER $NB_USER

ENTRYPOINT ["/usr/local/bin/tini", "--"]
CMD ["start-notebook.sh"]
