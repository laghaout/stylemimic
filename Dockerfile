FROM python:3.12-slim

# Declare the working directory.
WORKDIR stylemimic

# Install Linux utilities.
RUN apt -y update
RUN apt -y upgrade
RUN apt -y install less
RUN apt -y install tree

# Install Python packages.
RUN python3 -m pip install --upgrade pip
RUN pip3 install --user --upgrade pip
RUN pip3 install openai==1.35.7
RUN pip3 install mistralai==0.4.2
RUN pip3 install numpy==1.26.4
RUN pip3 install pandas==2.2.2
RUN pip3 install pytest==7.4.0
RUN pip3 install python-dotenv==1.0.1
RUN pip3 install pydantic==2.8.2
RUN pip3 install tiktoken==0.7.0

# Install the local package.
COPY Dockerfile *.yml *.txt *.sh *.toml README.* .pre-commit-config.yaml .env .env.local ./
COPY stylemimic/ stylemimic
COPY tests/ tests

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "main.py"]
