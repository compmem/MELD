FROM rpy2/jupyter

USER root

RUN apt-get update && apt-get install -y python3-tk

RUN pip3 install scipy matplotlib joblib

COPY . /home/jupyteruser/work/meld

RUN pip3 install /home/jupyteruser/work/meld

RUN /usr/local/bin/ipython -c "import meld"

RUN chown -R $NB_USER:users /home/jupyteruser/work/meld

USER $NB_USER

ENTRYPOINT ["/usr/local/bin/tini", "--"]
CMD ["start-notebook.sh"]