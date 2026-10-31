#!/bin/bash
# Adds "DoorCam Admin" to the Jetson's app menu and desktop. Run once as the desktop user:
#   ./install_admin_app.sh
set -e
APP_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! python3 -c "import tkinter" 2>/dev/null; then
    sudo apt-get install -y python3-tk
fi

ENTRY="[Desktop Entry]
Type=Application
Name=DoorCam Admin
Comment=Assign, revoke and restore door cards; camera feed
Exec=python3 $APP_DIR/admin_app.py
Path=$APP_DIR
Icon=preferences-desktop-user
Terminal=false
Categories=Utility;"

mkdir -p "$HOME/.local/share/applications"
echo "$ENTRY" > "$HOME/.local/share/applications/doorcam-admin.desktop"
if [ -d "$HOME/Desktop" ]; then
    echo "$ENTRY" > "$HOME/Desktop/doorcam-admin.desktop"
    chmod +x "$HOME/Desktop/doorcam-admin.desktop"
    gio set "$HOME/Desktop/doorcam-admin.desktop" metadata::trusted true 2>/dev/null || true
fi
echo "Installed. Open 'DoorCam Admin' from the app menu or the desktop icon."
