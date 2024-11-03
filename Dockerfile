# Use the official Python 3.11 image
FROM tensorflow/tensorflow:latest-gpu-jupyter
# Install Jupyter Notebook and additional data science libraries
RUN pip install --no-cache-dir notebook numpy pandas matplotlib scikit-learn tensorflow==2.15.0
