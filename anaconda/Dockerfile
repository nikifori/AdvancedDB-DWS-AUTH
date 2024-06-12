# Use the official Anaconda3 image as the base image
FROM continuumio/anaconda3

# Set the working directory
WORKDIR /home/advanced_db

# Copy all files from the current directory to the working directory in the container
COPY ./*.* /home/advanced_db/
COPY ./template /home/advanced_db/template

# Verify copy
RUN var=$(ls /home/advanced_db/)
RUN echo "$var"

# Create the environment and install the packages
RUN conda create --name advanced_db_env python=3.10.12 -y && \
    echo "source activate advanced_db_env" > ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc" && \
    conda run -n advanced_db_env pip install -r /home/advanced_db/requirements.txt

# Expose port 8000
EXPOSE 8000

# Run the Python scripts
ENTRYPOINT ["/bin/bash", "-c", "source activate advanced_db_env && python ingest_data.py && python backend.py"]
