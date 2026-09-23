#!/bin/sh
set -e

: "${EXTERNAL_IP:?EXTERNAL_IP must be set (the host LAN/public IP phones connect to)}"
: "${SIP_PASSWORD_1001:?}" "${SIP_PASSWORD_1002:?}" "${ARI_USER:?}" "${ARI_PASSWORD:?}"

for f in /etc/asterisk-templates/*.conf; do
  envsubst < "$f" > "/etc/asterisk/$(basename "$f")"
done

chown -R asterisk:asterisk /etc/asterisk /var/lib/asterisk /var/spool/asterisk /var/log/asterisk
exec asterisk -f -U asterisk -G asterisk -vvv
