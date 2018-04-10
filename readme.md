Server setup:

    sudo apt install nginx uwsgi uwsgi-plugin-python3 supervisor
    sudo apt install python3-pip python3-psycopg2
    sudo pip3 install sqlalchemy pillow flask

Configs:

    sudo vi /etc/uwsgi/apps-enabled/app.ini
    sudo vi /etc/nginx/sites-enabled/app.conf
    sudo vi /etc/supervisor/conf.d/app.conf