# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
FROM igibson/igibson:latest

RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    psmisc \
    vim \
    wget \
    curl \
    openssh-client \
    libglib2.0-0 \
    htop \
    less \
    && rm -rf /var/lib/apt/lists/*

# fzf
RUN git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
RUN ~/.fzf/install --all

WORKDIR /workspaces/
COPY requirements.txt .
RUN pip3 install -r /workspaces/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

WORKDIR /workspaces/scene-llm
CMD bash
