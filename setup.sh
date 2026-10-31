#!/bin/bash
# One-time DoorCam setup for the original Jetson Nano (JetPack 4.6 / L4T R32, Ubuntu 18.04).
# Run as your normal user from the doorcam folder:   bash setup.sh
# Takes ~1.5 hours (dlib and jetson-inference are compiled). Safe to re-run; finished steps are skipped.
set -e
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
TZ_NAME="Asia/Kolkata"
step() { echo; echo "======== $*"; }

if [ "$(id -u)" -eq 0 ]; then
    echo "Run this as your normal user (it uses sudo where needed), not as root."
    exit 1
fi
grep -q "R32" /etc/nv_tegra_release 2>/dev/null || echo "Warning: this doesn't look like JetPack 4.x, continuing anyway."
chmod +x "$APP_DIR"/*.sh "$APP_DIR"/*.py "$APP_DIR"/system/*.sh

step "System packages"
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev libpython3-dev python3-numpy python3-yaml \
    cmake git curl chrony libopenblas-dev liblapack-dev libjpeg-dev \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    python3-tk alsa-utils
sudo usermod -aG video "$USER"

step "Max performance power mode"
sudo nvpmodel -m 0 || true

step "4 GB swap file (the compiles below run out of memory without it)"
if ! swapon --show | grep -q /swapfile; then
    [ -f /swapfile ] || sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
fi
grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

step "Clock: timezone $TZ_NAME + chrony + boot-time sync check"
sudo timedatectl set-timezone "$TZ_NAME"
sudo systemctl disable --now systemd-timesyncd 2>/dev/null || true
CHRONY_CONF=/etc/chrony/chrony.conf
# Step the clock at any time, however far off it is (default only allows it in the first 3 updates).
if grep -q '^makestep' "$CHRONY_CONF"; then
    sudo sed -i 's/^makestep.*/makestep 1 -1/' "$CHRONY_CONF"
else
    echo 'makestep 1 -1' | sudo tee -a "$CHRONY_CONF"
fi
if ! grep -q 'doorcam' "$CHRONY_CONF"; then
    printf '\n# doorcam: extra time sources\npool time.google.com iburst maxsources 2\npool in.pool.ntp.org iburst maxsources 2\n' \
        | sudo tee -a "$CHRONY_CONF"
fi
sudo systemctl enable chrony
sudo systemctl restart chrony
sudo install -m 755 "$APP_DIR/system/doorcam-timefix.sh" /usr/local/bin/doorcam-timefix.sh
sudo install -m 644 "$APP_DIR/system/doorcam-timefix.service" /etc/systemd/system/doorcam-timefix.service
sudo systemctl daemon-reload
sudo systemctl enable doorcam-timefix.service
sudo systemctl restart doorcam-timefix.service

step "dlib with CUDA + face_recognition (compiling dlib takes ~1 hour)"
export PATH=/usr/local/cuda/bin:$PATH
python3 -m pip install --user --upgrade "pip<22" "setuptools<60" wheel
if python3 -c "import dlib, sys; sys.exit(0 if dlib.DLIB_USE_CUDA else 1)" 2>/dev/null; then
    echo "dlib with CUDA already installed"
else
    python3 -m pip install --user --no-cache-dir --verbose dlib==19.24.0
fi
python3 -m pip install --user "Pillow<9" "Click<8.1" face_recognition_models==0.3.0
python3 -m pip install --user --no-deps face_recognition==1.3.0
python3 -c "import dlib; print('dlib', dlib.__version__, 'CUDA:', dlib.DLIB_USE_CUDA)"

step "jetson-inference person detector (~30 min)"
if python3 -c "import jetson.inference" 2>/dev/null || python3 -c "import jetson_inference" 2>/dev/null; then
    echo "jetson-inference already installed"
else
    [ -d "$HOME/jetson-inference" ] || git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference "$HOME/jetson-inference"
    mkdir -p "$HOME/jetson-inference/build"
    cd "$HOME/jetson-inference/build"
    cmake -DBUILD_INTERACTIVE=NO ..
    make -j"$(nproc)"
    sudo make install
    sudo ldconfig
    cd "$APP_DIR"
fi

step "Downloading the detection model and building its TensorRT engine (first time only, a few minutes)"
python3 -c "
try:
    from jetson_inference import detectNet
except ImportError:
    from jetson.inference import detectNet
detectNet('ssd-mobilenet-v2', threshold=0.5)
print('detector ready')"

step "Autostart: replace the old cam.sh launcher with DoorCam"
mkdir -p "$HOME/.config/autostart" "$APP_DIR/known_faces" "$APP_DIR/logs"
OLD_REF='(^|[/ "=])cam\.sh'
for f in "$HOME"/.config/autostart/*.desktop; do
    [ -e "$f" ] || continue
    [ "$(basename "$f")" = "doorcam.desktop" ] && continue
    if grep -Eq "$OLD_REF" "$f"; then
        mv "$f" "$f.disabled"
        echo "disabled old autostart entry: $f"
    fi
done
if crontab -l 2>/dev/null | grep -Eq "$OLD_REF"; then
    crontab -l > "$HOME/crontab.before-doorcam"
    crontab -l | sed -E "\\#$OLD_REF# s/^/# disabled by doorcam: /" | crontab -   # (# delimiter: the pattern has a /)
    echo "disabled cam.sh in your crontab (backup: ~/crontab.before-doorcam)"
fi
OTHER=$(grep -lE "$OLD_REF" /etc/xdg/autostart/*.desktop /etc/systemd/system/*.service /etc/rc.local \
    "$HOME/.profile" "$HOME/.bashrc" 2>/dev/null || true)
if [ -n "$OTHER" ]; then
    echo "!! cam.sh is also started from these files; disable it there by hand so it doesn't fight DoorCam for the camera:"
    echo "$OTHER"
fi
cat > "$HOME/.config/autostart/doorcam.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=DoorCam
Exec=/bin/bash $APP_DIR/run_doorcam.sh
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=5
EOF

step "Done"
timedatectl | sed -n '1,6p'
echo
echo "Next:"
echo "  1. Plug in the USB card reader and run:  bash setup_storage.sh"
echo "  2. Enroll faces:                         python3 enroll.py --help"
echo "  3. Reboot. DoorCam starts on the monitor after login."
