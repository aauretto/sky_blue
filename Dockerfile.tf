# TO BE RUN ON THE HPC. WILL IMMEDIATELY EXECUTE ./src/main.py when run

FROM tensorflow/tensorflow:2.15.0-gpu

# Install certificate for online sources for pireps
COPY pirep_ssl_cert.crt /usr/local/share/ca-certificates/pirep_ssl_cert.crt
RUN update-ca-certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Needed for HPC -- Lets tensorflow fallback to suboptimal GPU algorithms rather
# than crash
ENV XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /skyblue
COPY . /skyblue

# Install pip requirements
RUN pip install -r requirements.txt