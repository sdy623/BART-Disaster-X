FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel
LABEL authors="syc"


COPY ./requirements.txt .
ENTRYPOINT ["top", "-b"]
RUN apt-get update -y && \
    apt-get install git -y --no-install-recommends && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118 && \
    rm -rf /usr/share/doc/* /usr/share/man/* /usr/share/locale/*