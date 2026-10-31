#!/bin/bash
# Started at login by ~/.config/autostart/doorcam.desktop. Restarts DoorCam if it crashes;
# pressing Q on the monitor exits cleanly and stops the restart loop.
cd "$(dirname "$(readlink -f "$0")")" || exit 1

# Only one copy at a time (v1 sometimes started several).
exec 9>/tmp/doorcam.lock
flock -n 9 || { echo "DoorCam is already running"; exit 0; }

# Keep the monitor on.
xset s off -dpms 2>/dev/null
gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null
gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null

mkdir -p logs
[ -f logs/console.log ] && [ "$(stat -c %s logs/console.log)" -gt 5000000 ] && mv logs/console.log logs/console.log.1

while true; do
    echo "=== $(date) starting DoorCam" >> logs/console.log
    python3 main.py >> logs/console.log 2>&1
    code=$?
    [ "$code" -eq 0 ] && break
    echo "=== $(date) DoorCam exited with code $code, restarting in 5 s" >> logs/console.log
    sleep 5
done
