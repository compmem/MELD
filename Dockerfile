FROM rpy2/jupyter

USER root

RUN apt-get update && apt-get install -y python3-tk build-essential libglu1

RUN pip3 --no-cache-dir install scipy scikit-learn matplotlib joblib==0.8.4 seaborn pandas tables statsmodels numba nilearn line_profiler Cython

COPY . /home/$NB_USER/meld

RUN pip3 --no-cache-dir install -e  /home/$NB_USER/meld

RUN /usr/local/bin/ipython -c "import meld"

RUN wget -q https://ftp.humanconnectome.org/workbench/workbench-linux64-v1.3.0.zip \
    && unzip workbench-linux64-v1.3.0.zip \
    && mv workbench /opt/workbench \
    && rm workbench-linux64-v1.3.0.zip && chown -R $NB_USER:users /opt/workbench

RUN chown -R $NB_USER:users /home/$NB_USER/meld

ENV JOBLIB_START_METHOD='forkserver'
ENV PATH=$PATH:/opt/workbench/bin_linux64
USER $NB_USER

ENTRYPOINT ["/usr/local/bin/tini", "--"]
CMD ["start-notebook.sh"]
