FROM rpy2/jupyter

USER root

RUN apt-get update && apt-get install -y python3-tk

RUN pip3 --no-cache-dir install scipy scikit-learn matplotlib joblib==0.8.4 seaborn pandas tables statsmodels numba nilearn 

COPY . /home/$NB_USER/work/meld

RUN pip3 --no-cache-dir install /home/$NB_USER/work/meld

RUN /usr/local/bin/ipython -c "import meld"

RUN mkdir -p /home/$NB_USER/code/nipype && git clone https://github.com/nipy/nipype /home/$NB_USER/code/nipype && pip3 install -e /home/$NB_USER/code/nipype

RUN chown -R $NB_USER:users /home/$NB_USER/

ENV JOBLIB_START_METHOD='forkserver'

USER $NB_USER

ENTRYPOINT ["/usr/local/bin/tini", "--"]
CMD ["start-notebook.sh"]