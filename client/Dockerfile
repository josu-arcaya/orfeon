FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DB_HOST=optimizer.cjxwnlyzdbn4.us-east-2.rds.amazonaws.com

RUN apt-get update && apt-get install -y \ 
    python3-pip \
    stress-ng \
    openssh-server

COPY files/requirements.txt /tmp/requirements.txt
RUN /usr/bin/pip3 install -r /tmp/requirements.txt

COPY files/harden.py /usr/local/bin/harden.py
#RUN /usr/bin/python3 /usr/local/bin/harden.py

COPY files/lynis-3.0.3.tar.gz /tmp/lynis-3.0.3.tar.gz
RUN tar zxf /tmp/lynis-3.0.3.tar.gz -C /opt/

COPY files/client.py /usr/local/bin/client.py

#CMD ["/usr/bin/python3", "/usr/local/bin/client.py", "-a"]
CMD /usr/bin/python3 /usr/local/bin/harden.py && /usr/bin/python3 /usr/local/bin/client.py -a
