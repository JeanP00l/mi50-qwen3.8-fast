# llama.cpp for AMD MI50 / Radeon VII (gfx906) with ROCm 7.14 and gfx906 kernel patches.
#
# Base: mixa3607/rocm-gfx906 - ROCm 7.x built with gfx906 support (TheRock).
# Official AMD ROCm 7.x dropped gfx906; on ROCm 6.3 `-sm tensor` hung about
# once per 8 long requests, on 7.14 - 0 of 30+.
#
# Build (from the repository root):
#   docker build -f docker/Dockerfile -t mi50-llama:b11235 .
ARG BASE=mixa3607/rocm-gfx906:7.14-complete
FROM ${BASE}

# Pinned llama.cpp tag: the patches are checked against it.
ARG LLAMA_TAG=b11235
# Set to 0 to build vanilla llama.cpp on the same base (for A/B comparison).
ARG PATCHES=1
# Mirror/fork override if GitHub is slow for you.
ARG LLAMA_REPO=https://github.com/ggml-org/llama.cpp.git

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential libcurl4-openssl-dev ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMA_TAG}" "${LLAMA_REPO}" . \
    && git rev-parse HEAD > /src/UPSTREAM_COMMIT

# Every patch checks its anchors and fails the build on a miss,
# so an image can never silently end up without a patch.
COPY patches/ /patches/
RUN echo "${LLAMA_TAG}" > /src/PATCHES \
    && if [ "${PATCHES}" = "1" ]; then \
        python3 /patches/gfx906-gdn-multicolumn.py ggml/src/ggml-cuda/gated_delta_net.cu \
        && python3 /patches/gfx906-mmvf-few-rows.py ggml/src/ggml-cuda/mmvf.cu \
        && python3 /patches/gfx906-mmvq-q4_0-multicol.py ggml/src/ggml-cuda/mmvq.cu \
        && echo "+gfx906-gdn-multicolumn +gfx906-mmvf-few-rows +gfx906-mmvq-q4_0-multicol" >> /src/PATCHES ; fi

ENV ROCM_PATH=/opt/rocm
# gfx906 = 9.0.6; harmless identity override in case a ROCm device list check trips.
ENV HSA_OVERRIDE_GFX_VERSION=9.0.6

RUN cmake -B build \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=gfx906 \
        -DGGML_HIP_GRAPHS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=ON \
        -DLLAMA_BUILD_TESTS=ON \
    && cmake --build build --config Release -j"$(nproc)" \
    && cmake --install build --prefix /usr/local \
    && cp build/bin/test-backend-ops /usr/local/bin/ \
    && rm -rf /src/build \
    && ldconfig

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/llama-server"]
