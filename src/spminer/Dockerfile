# Use the older Python image
FROM python:3.7.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --- FIX FOR DEBIAN BUSTER EOL (404 Errors) ---
# Debian 10 is EOL, so we must point apt to the archive servers.
# We also disable the "valid-until" check because archive snapshots are old.
RUN echo "deb http://archive.debian.org/debian buster main" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security buster/updates main" >> /etc/apt/sources.list && \
    echo "Acquire::Check-Valid-Until false;" > /etc/apt/apt.conf.d/99no-check-valid-until

# --- INSTALL SYSTEM DEPENDENCIES ---
# Now apt-get update will work.
# We need build-essential (gcc) for compiling numpy/torch-cluster
# We need libgomp1 for OpenMP support in PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# --- PYTHON DEPENDENCIES ---

# 1. Upgrade pip (older pip versions fail to find modern wheels)
RUN pip install --upgrade pip

# 2. Install Cython and Numpy FIRST.
# Installing these before other packages prevents build failures.
# RUN pip install --no-cache-dir Cython numpy==1.13.3

# 3. Install PyTorch CPU version explicitly.
RUN pip install --no-cache-dir torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html
RUN pip install torch-sparse==0.6.1 -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html
RUN pip install torch-cluster==1.5.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html
RUN pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html

# 4. Install the rest of the requirements.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 5. Setup the permissions
RUN chmod 1777 data_preprocessed_batch

## select either "t" or "st" 
CMD ["python", "batch_process.py", "t"]