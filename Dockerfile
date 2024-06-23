# Comment in to use the GPU. (Remember to use the `--gpus all` flag with `docker run`.)
#FROM tensorflow/tensorflow:2.15.0-gpu

# Alternatively, use the CPU-version. Choose either one of the two lines below. If choosing the base image for TensorFlow, comment out the pip3 installation of TensorFlow further down.
#FROM tensorflow/tensorflow:2.15.0
FROM python:3.11-slim

# Declare the working directory.
WORKDIR /stylemimic

# Set the environment variables.
ENV DOCKERIZED Yes

# Install Linux utilities.
RUN apt -y update
RUN apt -y upgrade
RUN apt -y install emacs-nox
RUN apt -y install less
RUN apt -y install tk
RUN apt -y install tree

# Install Python packages.
RUN python3 -m pip install --upgrade pip
RUN pip3 install --user --upgrade pip
RUN pip3 install tensorflow==2.15.0
RUN pip3 install sentence_transformers==2.5.1
RUN pip3 install seaborn==0.12.2
RUN pip3 install matplotlib==3.8.0
RUN pip3 install numpy==1.26.4
RUN pip3 install pandas==2.1.4
RUN pip3 install scikit-learn==1.2.2
RUN pip3 install pytest==7.4.0
RUN pip3 install py-cpuinfo==9.0.0
RUN pip3 install python-dotenv==1.0.1
RUN pip3 install pydantic==1.10.12

# Install the local package.
COPY Dockerfile *.yml *.txt *.sh *.toml README.* .pre-commit-config.yaml .env ./
COPY stylemimic/ stylemimic
COPY tests/ tests

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "stylemimic/learner.py"]
