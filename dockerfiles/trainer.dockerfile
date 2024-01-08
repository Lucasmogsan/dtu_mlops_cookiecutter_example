# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Application (copy necessary files, COPY <src-path> <destination-path>):
# Can copy all by using COPY . . (but will copy all files, including the ones we don't need)
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY dtu_mlops_cookiecutter_example/ dtu_mlops_cookiecutter_example/
COPY data/ data/

# Set the working directory
WORKDIR /
# install the package but disable the cache
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir


# Name entrypoint (what to run when the container starts) - "u" for unbuffered output (to our terminal)
# Change the entrypoint to run whatever you want (now it runs the train_model.py script with the "train" argument)
ENTRYPOINT ["python", "-u", "dtu_mlops_cookiecutter_example/train_model.py", "train"]