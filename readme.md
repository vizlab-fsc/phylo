Requires [`faiss`](https://github.com/facebookresearch/faiss/). To install (Ubuntu 16.04):

    sudo apt install libopenblas-dev python3-numpy python3-dev
    git clone git@github.com:facebookresearch/faiss.git
    cd faiss
    cp example_makefiles/makefile.inc.Linux makefile.inc

    vi makefile.inc
    # then look for the line "This is for Centos:" and comment out the BLASLDFLAGS? line.
    # uncomment the one for Ubuntu 16.
    # then look for the PYTHONCFLAGS line and change 2.7 to 3.5 (or whatever is the correct python version)

    # build
    make

    # build the python interface
    # this results in `faiss.py`
    make py