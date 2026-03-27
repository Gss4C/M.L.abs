FROM python:3.11-slim
WORKDIR /workspace
RUN pip install jupyterlab pandas matplotlib
EXPOSE 8888
CMD ["jupyter-lab", "--port=8888", "--ip=0.0.0.0", "--no-browser", "--allow-root","--NotebookApp.token=''", "--NotebookApp.password=''"]