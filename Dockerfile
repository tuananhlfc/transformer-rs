FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

# Install the curl and build-essential packages
RUN apt-get update && \
    apt-get install -y curl build-essential && \
    apt-get clean

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add cargo to the PATH for all users
ENV PATH="/root/.cargo/bin:${PATH}"
