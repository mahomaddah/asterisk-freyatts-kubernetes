FROM ubuntu:24.04

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      asterisk asterisk-core-sounds-en-wav gettext-base \
 && rm -rf /var/lib/apt/lists/*

# Our config files are templates rendered over the package defaults at start-up
# (secrets and the public address are filled in from the environment).
COPY config/ /etc/asterisk-templates/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5060/udp 10000-10099/udp
ENTRYPOINT ["/entrypoint.sh"]
