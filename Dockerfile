# Set base image
FROM python:3.10

# Get installations, python, etc. 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    ca-certificates \
    git \
    python3-pip \
    python3-dev \
    tcsh \
    perl \
    gcc \
    dc \
    bc \
    libgomp1 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff-dev \
    libfftw3-dev \
    libeigen3-dev \
    libgsl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install FSL
RUN wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py -O /tmp/fslinstaller.py && \
    python3 /tmp/fslinstaller.py -d /usr/local/fsl && \
    rm /tmp/fslinstaller.py

# Set the environment variables
ENV FSLDIR=/usr/local/fsl
ENV PATH="${FSLDIR}/bin:${PATH}"
ENV FSLOUTPUTTYPE=NIFTI_GZ

# Make working directory and environment path
RUN mkdir -p /app
WORKDIR /app
ENV PYTHONPATH=/app

# Copy and install requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Make the pipeline scripts executable
RUN chmod +x ./scripts/pipeline.py

# Entrypoint
ENTRYPOINT ["./scripts/pipeline.py"]