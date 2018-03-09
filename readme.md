Server setup:

    sudo apt install gcc gfortran build-essential g++ make cmake autoconf swig
    sudo apt install nginx uwsgi uwsgi-plugin-python3 supervisor
    sudo apt install python3-pip python3-psycopg2
    sudo pip3 install sqlalchemy pillow flask

Configs:

    sudo vi /etc/uwsgi/apps-enabled/app.ini
    sudo vi /etc/nginx/sites-enabled/app.conf
    sudo vi /etc/supervisor/conf.d/app.conf

Requires [`faiss`](https://github.com/facebookresearch/faiss/). To install (Ubuntu 16.04):

    sudo apt install libopenblas-dev python3-numpy python3-dev
    git clone git@github.com:facebookresearch/faiss.git
    cd faiss
    cp example_makefiles/makefile.inc.Linux makefile.inc

    vi makefile.inc
    # then look for the line "This is for Centos:" and comment out the BLASLDFLAGS? line.
    # uncomment the one for Ubuntu 16.

    # then look for the PYTHONCFLAGS line and change 2.7 to 3.5 (or whatever is the correct python version)
    # e.g.
    # PYTHONCFLAGS=-I/usr/include/python3.5/ -I/usr/lib/python3/dist-packages/numpy/core/include/

    # build
    make

    # build the python interface
    # this results in `faiss.py`
    make py

Then move all this stuff into the `faiss` folder so it imports correctly