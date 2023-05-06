FROM python:3.9

RUN pip install --upgrade pip

# Create user and set ownership and permissions as required
RUN useradd -ms /bin/bash roland
RUN mkdir /home/roland/app/ && chown -R roland:roland /home/roland/app
WORKDIR /home/roland/app/
USER roland

ENV PATH="/home/roland/.local/bin:${PATH}"

COPY --chown=roland:roland . .

RUN pip install --user --requirement requirements.txt

CMD [ "/bin/sh", "-c"]