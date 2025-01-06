# HPDGA Fall 24

Welcome to the **HPDGA Fall 24** repository!

## Description

This repository contains the source code for the HPDGA Fall 24 challenge. The code is a parallel implementation of the Hierarchical Navigable Samall World (HNSW) algorithm using CUDA. The HNSw is an Aproximate K Nearest Neighbors search algorithm and is highly parallelizable, as shown by the performance gains. 
## Sections

- Run it locally
- Run it remotely on Google Colab

### Run it locally

1. **Clone the repository:**
   If you are reading this section, it means you have access to this private repository. Clone the repository to your local machine using the following command:

   ```bash
    git clone https://github.com/vitorbborges/hpdga-fall24.git
    cd hpdga-fall24
    ```

2. **Install the dependencies:**
   To properly run and profile everything, you need to install nvidia tools. You can do this by running the following command:

   ```bash
    set -x \
    && cd $(mktemp -d) \
    && wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run \
    && sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit \
    && rm cuda_12.1.0_530.30.02_linux.run
    ```

3. **Run the project:**

After cloning the repository, to compile and run the project locally you need to inform the NVCC compiler what is the CUDA architecture of your GPU. To do this, you need to set the `CUDA_ARCH` environment variable to the architecture number of your GPU. You can refer to the [Nvidia Developer](https://developer.nvidia.com/cuda-gpus portal to find it.

Note: It is also possible to just run `nvidia-smi` on your terminal and ask ChatGPT what the architecture number should be. DO THIS AT YOUR OWN RISK. After finding out the number, run the following on terminal:

```bash
touch .env
echo "CUDA_ARCH={YOUR ARCHITECTURE NUMBER GOES HERE}" > .env
```

Now you can compile and run the project:

    ```bash
    # Compile the program
    !mkdir -p build \
    && cd build \
    && cmake .. \
    && cmake --build .
    ```

Run the program:

    ```bash
    cmake --build . --target run
    ```


### Run it remotely on Google Colab

1. **Make sure you received the .env file:**
   To run the project on Google Colab, you need to have the `.env` file with the GITHUB_TOKEN and CUDA_ARCH variables. Make sure you have received this file from the project owner. This is important to seemlesly clone the repository and inform the compiler what is the GPU architecture of the T4 GPU currently available on Google Colab.

2. **Run the project:**

After receiving the `.env` file, you can take a look into `cuda_colab.ipynb` that was sent togheter with the `.env` file. This notebook needs to be placed in the same Google Drive folder as the `.env` file. 

You also need to edit the path to the `.env` file in the notebook. This is the last cell of the notebook that needs editing. After that, you can run the notebook and it will clone the repository, compile, run and profile everything for you.        

