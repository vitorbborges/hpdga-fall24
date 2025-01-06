# High-Performance Data & Graph Analytics - Fall 2024

Repository for the High-Performance Data and Graph Analytics contest.

Main deadlines:

- Register for the contest: 2024-11-21 (yy-mm-dd) @ 23:59 CET
- Submission (as detailed in the lecture's slides): 2024-12-30 (yy-mm-dd) @ 23:59 CET (NO LATE COMMITS!)

For further details follow the information on the lecture's slides. 

## Contest instructions

The contest is organized in two main steps:

1. Accelerate the search procedure for varying values of K (K=100, K=1000)
2. [Optional - Only after 1 is completed] Accelerate the index construction step

You are responsible for completing the two steps and properly organizing the time at your disposal!

Create a [release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) once step-1 is completed.

### Build and run
You can build and execute the existing implementation by running the following commands:

```sh
mkdir build && cd build
cmake ..
make
./hnsw
```

In the repository, you'll find the smallest dataset. To download the standard one: 

```sh
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xvzf sift.tar.gz
```

You will have to modify the `Makefile` in order to compile the code with `nvcc`, as seen during the lectures.

###  How to use Colab
1) From your *private copy of this repository*, open the `colab.ipynb` file in this folder and click on the "open in Colab" button
2) Create a GihHub fine-grained personal access token just for this repository [link](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) (remember to give the right permission to the token in order to use it)
3) Select the `GPU runtime` in Colab [link](https://www.geeksforgeeks.org/how-to-use-google-colab)
4) Change the parameters accordingly with your account ID, clone repository name and token

We suggest editing the original code on your local machine, commit it on GitHub and then load the changes inside of your Colab environment by executing the cell for pulling the remote commits.

## Credits

This is a highly simplified and didactic version of https://github.com/arailly/hnsw with additional comments and parsing of .ivec/.fvec input files.


