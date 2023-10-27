################################################################################
### GENERAL SETUP
################################################################################

FROM ubuntu:22.04 as llvm

# User directory
ENV USER=user
ENV HOME=/home/user
WORKDIR $HOME

# Install dependencies
RUN apt-get update -y && \ 
  apt-get install -y --no-install-recommends \
  wget=1.21.2-2ubuntu1 \
  ca-certificates=20230311ubuntu0.22.04.1 \
  git=1:2.34.1-1ubuntu1.9 \
  clang=1:14.0-55~exp2 \
  lld=1:14.0-55~exp2 \
  cmake=3.22.1-1ubuntu1.22.04.1 \
  ninja-build=1.10.1-1 \
  ccache=4.5.1-1 \
  python3=3.10.6-1~22.04 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Clone project
# RUN git clone --recurse-submodules --depth 1 --shallow-submodules https://github.com/Berke-Ates/MLIR-Smith
COPY . $HOME/MLIR-Smith/
RUN mkdir $HOME/bin
WORKDIR $HOME/MLIR-Smith

################################################################################
### Install LLVM/MLIR with MLIR-Smith
################################################################################

# Build LLVM/MLIR
WORKDIR $HOME/MLIR-Smith/llvm-project-smith/build
RUN rm -rf ./*

RUN cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_CCACHE_BUILD=ON \ 
  -DLLVM_USE_SANITIZER="Address;Undefined" \
  -DLLVM_INSTALL_UTILS=ON && \
  ninja && \
  cp $HOME/MLIR-Smith/llvm-project-smith/build/bin/mlir-opt $HOME/bin/ && \
  cp $HOME/MLIR-Smith/llvm-project-smith/build/bin/mlir-translate $HOME/bin/ && \
  cp $HOME/MLIR-Smith/llvm-project-smith/build/bin/mlir-smith $HOME/bin/

# Go home
WORKDIR $HOME

################################################################################
### Install MLIR-DaCe with SDFG-Smith
################################################################################

# Build MLIR-DaCe
WORKDIR $HOME/MLIR-Smith/mlir-dace-smith/build
RUN rm -rf ./*

RUN cmake -G Ninja .. \
  -DMLIR_DIR="$PWD/../../llvm-project-smith/build/lib/cmake/mlir" \
  -DLLVM_EXTERNAL_LIT="$PWD/../../llvm-project-smith/build/bin/llvm-lit" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_USE_SANITIZER="Address;Undefined" && \
  ninja && ninja sdfg-smith && \
  cp $HOME/MLIR-Smith/mlir-dace-smith/build/bin/sdfg-opt $HOME/bin/ && \
  cp $HOME/MLIR-Smith/mlir-dace-smith/build/bin/sdfg-translate $HOME/bin/ && \
  cp $HOME/MLIR-Smith/mlir-dace-smith/build/bin/sdfg-smith $HOME/bin/

# Go home
WORKDIR $HOME

################################################################################
### Reduce Image Size
################################################################################

# Copy binaries
FROM ubuntu:22.04

# User directory
ENV USER=user
ENV HOME=/home/user
WORKDIR $HOME

# Move dotfiles
RUN mv /root/.bashrc . && mv /root/.profile .

# Make terminal colorful
ENV TERM=xterm-color

# Install dependencies
RUN apt-get update -y && \ 
  apt-get install -y --no-install-recommends \
  emscripten=3.1.5~dfsg-3ubuntu1 \
  wabt=1.0.27-1 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Launch bash shell at home
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["cd $HOME && bash"]

# Copy Binaries
RUN mkdir $HOME/bin
COPY --from=llvm $HOME/bin/* $HOME/bin/

# Add binaries to PATH
ENV PATH=$HOME/bin:$PATH

################################################################################
### Copy files over
################################################################################

COPY ./docs ./docs
COPY ./scripts ./scripts
COPY ./README.md ./README.md
COPY ./LICENSE ./LICENSE
