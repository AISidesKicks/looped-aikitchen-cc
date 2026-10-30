# loop-llama-cpp: llama.cpp with NVIDIA Dory support (PR 30306)
#
# The official ghcr.io/ggml-org llama.cpp release images do not know the
# "dory" GGUF architecture yet (verified 2026-10-30, v0.8.0). So we build
# our own from the PR head, pinned by commit. When the PR merges upstream
# and a release tag ships it, drop this Dockerfile and point compose back
# at the official image.

ARG CUDA_TAG="13.1.2-devel-ubuntu24.04"

FROM nvidia/cuda:${CUDA_TAG} AS build

# pinned: ggml-org/llama.cpp PR 30306 head, fork brianchmielNV/llama.cpp
# update the sha together with the note above when moving to a new head
ARG LLAMA_HEAD="1c3cdd23787750814aa21a070dc4d6dc451af871"

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential ccache libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# RTX 4070 is Ada Lovelace (sm_89); restrict arch to keep the build fast
ARG GGML_CUDA_ARCH="89"
WORKDIR /src

RUN git clone --depth 1 https://github.com/brianchmielNV/llama.cpp /src \
    && cd /src \
    && git fetch --depth 1 origin ${LLAMA_HEAD} \
    && git checkout --detach ${LLAMA_HEAD} \
    && echo "pinned" > /HEAD && git rev-parse HEAD

# link the CUDA driver API against the toolkit stubs (no real driver inside the build container)
ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}"

RUN cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=${GGML_CUDA_ARCH} \
      -DLLAMA_CURL=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DGGML_OPENMP=OFF \
    && cmake --build build --target llama-server -j"$(nproc)" \
    && rm -rf build/.cmake* /tmp/*

FROM nvidia/cuda:${CUDA_TAG}

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libcurl4t64 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig \
    && apt-get autoremove -y

COPY --from=build /src/build/bin/llama-server /usr/local/bin/llama-server

ENTRYPOINT ["/usr/local/bin/llama-server"]
