#!/bin/bash
# Runs at boot (doorcam-timefix.service). Gets the clock right, then creates the flag file
# DoorCam checks before it files footage under a date.
# 1) normal NTP via chrony; 2) if NTP is blocked on this network, use the Date header
# of an HTTPS server instead (-k because certificate checks fail while the clock is wrong).
FLAG=/run/doorcam-time-ok
rm -f "$FLAG"
while true; do
    if chronyc waitsync 6 0 0 5 >/dev/null 2>&1; then
        touch "$FLAG"
        logger -t doorcam-timefix "clock synchronised by NTP"
        exit 0
    fi
    for url in https://www.google.com https://www.cloudflare.com; do
        d=$(curl -skI --max-time 10 "$url" | tr -d '\r' | sed -n 's/^[Dd]ate: //p' | head -n1)
        if [ -n "$d" ] && date -s "$d" >/dev/null 2>&1; then
            touch "$FLAG"
            logger -t doorcam-timefix "NTP unavailable, clock set from $url: $d"
            exit 0
        fi
    done
    logger -t doorcam-timefix "no internet time source yet, retrying in 30 s"
    sleep 30
done
