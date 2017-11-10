FROM rpy2/jupyter

USER root

RUN apt-get update && apt-get install -y python3-tk

RUN pip3 --no-cache-dir install scipy matplotlib joblib==0.8.4 seaborn pandas tables statsmodels numba

COPY . /home/$NB_USER/work/meld

RUN pip3 --no-cache-dir install /home/$NB_USER/work/meld

RUN /usr/local/bin/ipython -c "import meld"

RUN chown -R $NB_USER:users /home/$NB_USER/work/meld

ENV JOBLIB_START_METHOD='forkserver'

USER $NB_USER

ENTRYPOINT ["/usr/local/bin/tini", "--"]
CMD ["start-notebook.sh"]