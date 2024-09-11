FROM qwenllm/qwen:1.5-cu121
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    wget
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
WORKDIR /app
CMD ["/bin/bash"]
