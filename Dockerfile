# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Set environment variables and time zone
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=America/New_York
ENV PYTHONPATH /app:$PYTHONPATH

# Set the time zone, non-interactively
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Update and install system dependencies
RUN apt-get update && \
    apt-get install -y wget git libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1

# Install dependencies for FFmpeg and libx264
RUN apt-get update && \
    apt-get install -y ffmpeg libx264-dev

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Add Conda to PATH
ENV PATH /opt/conda/bin:$PATH

# Create a new Conda environment and install Python
RUN conda create -n myenv python=3.10 -y
ENV PATH /opt/conda/envs/myenv/bin:$PATH
ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
# Set working directory
WORKDIR /app 

# Activate the Conda environment and install PyTorch with CUDA support
RUN /bin/bash -c "source activate myenv && \
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"

RUN /bin/bash -c "source activate myenv && \
    pip install git+https://github.com/huggingface/transformers"
# Install necessary Python packages for the application
RUN /bin/bash -c "source activate myenv && \
    pip install streamlit librosa soundfile"

# Install Pyannote.audio
RUN /bin/bash -c "source activate myenv && \
    pip install pyannote.audio flash_attn"


# Copy the Streamlit app into the container
COPY . .

# Install dependencies from the repository
RUN /bin/bash -c "source activate myenv && \
    pip install -r requirements.txt"


# Expose the Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["bash", "-c", "source activate myenv && streamlit run app.py"]

