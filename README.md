# DoorCam: secure door access with an RFID card + face verification

A door camera system for the **NVIDIA Jetson Nano** that lets a person into a room only when
their **RFID card** and their **live face** both check out.

1. The person taps their card. The system loads **only that card owner's** face data.
2. The outside speaker says *"Hi \<name\>, please look at the camera"* once the face matches.
3. It asks for a **random challenge** (e.g. *"Blink twice"*, *"Show a thumbs up"*), so a photo
   or a recorded video can't get in.
4. If the face matches on every check and the challenge is passed, the door opens (a servo
   presses the existing exit sensor). Otherwise the inside monitor alerts someone.

It also keeps the **version 2** features: all-day recording, person clips, and name boxes on the
inside monitor. An **admin app** manages cards and people.

| Part | What it does |
|---|---|
| `admin_app.py` | Admin app: Cards, Assign, Revoke, Restore, Delete, full-screen Camera feed with face recognition |
| `liveness_demo.py` | Entry demo on a laptop: card tap → face match → greeting → random challenge → granted/denied |
| `register_face.py` | Register a face from the command line (the admin app does the same) |
| `users.py` | Card registry from the command line (`list`, `assign`, `revoke`, `restore`, `delete`) |
| `main.py` | DoorCam v2 on the Jetson: live monitor, person detection, recording |
| `verify.py`, `recognizer.py` | Face recognition (ArcFace) for the entry check and the live view |
| `liveness.py` | The random challenge (blinks, head turns, hand gestures) |
| `registration.py`, `facepoints.py` | Guided face registration (like Face ID) with voice prompts |
| `prompts/`, `make_prompts.py` | Spoken prompts (generated with the Piper voice model) |
| `docs/` | Jetson guide, start-on-boot guide, one-page project plan |

There are two ways to run it:

- **[Part A](#part-a-try-it-on-a-laptop-about-20-minutes)**: on any laptop with a webcam
  (Windows, macOS or Linux). This is the easiest way to demonstrate it.
- **[Part B](#part-b-set-it-up-on-the-jetson-nano)**: on the Jetson Nano at the door.

---

## Part A: try it on a laptop (about 20 minutes)

**You need:** a laptop with a webcam and speakers, internet, about 1.5 GB of free disk space,
**Python 3.9 to 3.12** (3.11 recommended) and **git**.

### Step 1. Install Python and git

| System | Commands |
|---|---|
| **Windows** | Install Python 3.11 from <https://www.python.org/downloads/> (tick **"Add python.exe to PATH"**) and Git from <https://git-scm.com/download/win>. |
| **macOS** | Install Python 3.11 from <https://www.python.org/downloads/macos/> (includes Tk, which the app needs). Git: run `xcode-select --install`. With Homebrew instead: `brew install python@3.11 python-tk@3.11 git`. |
| **Ubuntu / Debian** | `sudo apt update && sudo apt install -y python3 python3-venv python3-pip python3-tk git` |

Check them:

```bash
python3 --version      # Windows: py -3.11 --version
git --version
```

### Step 2. Download the project

```bash
git clone https://github.com/tadiPro250/doorcam.git
cd doorcam
```

### Step 3. Create a virtual environment and install the libraries

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-laptop.txt
```

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-laptop.txt
```

The libraries it installs (`requirements-laptop.txt`):

| Library | Used for |
|---|---|
| `numpy<2` | arrays and maths (MediaPipe 0.10.14 needs NumPy 1.x) |
| `opencv-python` | camera, image processing, face alignment |
| `pyyaml` | reading `config.yaml` |
| `onnxruntime` | running the ArcFace face-recognition model |
| `mediapipe==0.10.14` | face points (eyes, nose, head angle) and hand gestures |
| `piper-tts` | generating the spoken prompts and the "Hi \<name\>" greetings |
| `tkinter` | the admin app window (comes with Python; on Ubuntu it is `python3-tk`) |

Keep the virtual environment active (`source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1`
on Windows) for all the steps below.

### Step 4. Download the models (once, about 330 MB)

```bash
python scripts/download_models.py
```

This saves into `models/`:

- **`w600k_r50.onnx`**: the ArcFace face-recognition model from InsightFace. It is licensed for
  non-commercial research use, so it is downloaded rather than included here.
- **`en_US-lessac-medium.onnx`** and its `.json`: the Piper voice.

The standard voice prompts are already in `prompts/`. To regenerate them (optional):
`python make_prompts.py --voice models/en_US-lessac-medium.onnx --force`

### Step 5. Open the admin app and register a person

```bash
python admin_app.py
```

1. Go to **Assign** and type a card ID, e.g. `DEMO0001` (a USB RFID reader types the ID by
   itself when a card is tapped). The app shows **New card**.
2. Choose **New person**, type the name and choose **Admin**. The first admin protects the app
   with a lock screen.
3. Press **Register face & assign**. The camera opens and a voice guides you:
   - Look straight at the camera.
   - Turn left, then right, then up, then down, following the dial on the right.
   - It only takes pictures while you hold still in the green box (20 pictures, about 30 seconds).
4. When it finishes, the person and card appear in **Cards**. Their "Hi \<name\>" greeting is
   generated automatically.

Then try the other tabs:

- **Assign** with the same card ID: it now shows **In use**, with options to change the access
  level, re-register the face, revoke the card or delete the person. Each person has exactly one
  card: giving someone a new card replaces their old one.
- **Revoke / Restore**: switch a card off and on.
- **Delete**: removes a person's card, face data and greeting, and shows the storage used.
- **Camera feed**: opens **full screen** with live face recognition. Registered people get a
  green box with their name, strangers a red "Unknown" box. **Esc** or **Q** closes it.
- **Lock** (or Ctrl+L): the app now asks for an admin card. Type `DEMO0001` and press Enter.

macOS asks for camera permission the first time: allow it for Terminal (or your editor), then
start the app again.

### Step 6. Run the entry demo (card + face + random challenge)

Close the admin app first, so the camera is free, then:

```bash
python liveness_demo.py --card DEMO0001
```

- **SPACE** = tap the card. You hear *"Please look at the camera"*, then *"Hi \<name\>…"* once
  your face matches, then two random actions such as *"Blink twice"*, *"Turn your head to your
  left"* or *"Show a thumbs up"*.
- Do each action after you hear it. The result is **Access granted** or **Access denied**.
- **U** = tap an unregistered card. **Q** = quit.
- Add `--hide-prompts` to hear the actions only, with no text on screen (like at the real door).

Things to try:

- **Someone else** in front of the camera: denied at once (*face does not match the card*).
- **A photo of the registered person** on a phone: it matches, but fails the blink and turn
  actions.

The bottom bar shows the live match score: about 0.6 to 0.8 for the right person, near 0 for
someone else.

### Command-line alternatives

```bash
python register_face.py "Ravi Kumar"          # register a face (guided, with voice)
python users.py assign DEMO0002 "Ravi Kumar"   # give them a card (add --admin for an admin)
python users.py list
python users.py revoke DEMO0002
python users.py restore DEMO0002
python users.py delete DEMO0002                # removes card, face data and greeting
```

### Troubleshooting (laptop)

| Problem | Fix |
|---|---|
| `No module named '_tkinter'` | Python was installed without Tk. Use the python.org installer, or `brew install python-tk@3.11`, or `sudo apt install python3-tk`. |
| `pip install mediapipe` fails | Use Python 3.9 to 3.12 (MediaPipe 0.10.14 has no build for newer versions). |
| Camera is black or busy | Close other apps that use the camera (Zoom, Teams, the admin app while the demo runs). On macOS, allow camera access in System Settings → Privacy & Security → Camera. |
| No sound | Check the volume. Linux needs `aplay` (`sudo apt install alsa-utils`). |
| "The face model could not be loaded" | Run `python scripts/download_models.py`. |

---

## Part B: set it up on the Jetson Nano

### Hardware

- Original **Jetson Nano** (4 GB) with **JetPack 4.6** (Ubuntu 18.04, Python 3.6). Use a 5 V 4 A
  barrel-jack power supply.
- USB webcam, inside monitor, keyboard and mouse.
- **32 GB SD card in a USB reader** for footage. This is a separate card, not the boot card.
- **USB speaker** (or USB sound card + amplifier) outside. The Nano has no headphone jack.
- Later: a 13.56 MHz RFID reader (PN532 / RC522 / ACR122U) and an Arduino Nano + SG90 servo for
  the lock. See `docs/Secure_Door_Access_Plan.docx`.

### Step 1. Download the project

```bash
sudo apt update && sudo apt install -y git
cd ~
git clone https://github.com/tadiPro250/doorcam.git
cd ~/doorcam
```

### Step 2. Run the setup script (about 1.5 hours, needs internet)

```bash
bash setup.sh
```

`setup.sh` installs and configures everything for DoorCam v2:

| What | Installed with |
|---|---|
| System packages | `sudo apt-get install python3-pip python3-dev libpython3-dev python3-numpy python3-yaml cmake git curl chrony libopenblas-dev liblapack-dev libjpeg-dev gstreamer1.0-plugins-good gstreamer1.0-plugins-bad python3-tk alsa-utils` |
| Max performance mode | `sudo nvpmodel -m 0` |
| 4 GB swap file | needed to compile dlib |
| Clock | timezone Asia/Kolkata, `chrony` for automatic time sync, and a boot service that also works when NTP is blocked |
| dlib with CUDA + face_recognition | `pip3 install --user "pip<22" "setuptools<60" wheel`, `dlib==19.24.0`, `"Pillow<9" "Click<8.1" face_recognition_models==0.3.0`, `face_recognition==1.3.0 --no-deps` |
| Person detector | `jetson-inference`, built from source (SSD-MobileNet-v2 on TensorRT) |
| Autostart | starts DoorCam when you log in, replacing the old `cam.sh` |

### Step 3. Install the extra pieces for the card + face check

**a) Models** (same script as on the laptop):

```bash
python3 scripts/download_models.py
```

**b) ONNX Runtime for the face-recognition model.** There is no regular `pip` package for
Python 3.6 on the Jetson. Download the **onnxruntime-gpu** wheel for **JetPack 4.6 /
Python 3.6** from NVIDIA's Jetson Zoo (<https://elinux.org/Jetson_Zoo#ONNX_Runtime>), then:

```bash
pip3 install --user ~/Downloads/onnxruntime_gpu-*-cp36-cp36m-linux_aarch64.whl
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

**c) Piper**, for "Hi \<name\>" greetings of newly registered people. Piper's Python package needs
Python 3.9 or newer, so on the Nano you use the standalone program:

```bash
cd ~
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
tar xzf piper_linux_aarch64.tar.gz
sudo ln -sf ~/piper/piper /usr/local/bin/piper
echo "test" | piper --model ~/doorcam/models/en_US-lessac-medium.onnx --output_file /tmp/t.wav && aplay /tmp/t.wav
cd ~/doorcam
```

**d) Speaker check:** `aplay -l` lists the sound devices; `aplay prompts/granted.wav` should
say "Access granted". If the USB speaker isn't the default device, use `aplay -D plughw:1,0`.

### Step 4. Footage card

```bash
bash setup_storage.sh                      # lists drives: find the 32 GB card (e.g. /dev/sda1)
bash setup_storage.sh /dev/sda1 --format   # ERASES it, formats ext4, mounts it at /footage on every boot
```

### Step 5. Admin app

```bash
./install_admin_app.sh                     # adds "DoorCam Admin" to the app menu and the desktop
python3 admin_app.py --fullscreen          # or start it by hand
```

Use it exactly as in Part A, step 5. On the Jetson it uses **dlib** for face points, because
MediaPipe doesn't run on Python 3.6. **Quit DoorCam first** (press **Q** on its monitor window)
so the camera is free.

### Step 6. Reboot

```bash
sudo reboot
```

After login, DoorCam starts on the monitor by itself. See **[docs/START_ON_BOOT.md](docs/START_ON_BOOT.md)**
for the full checklist and **[docs/JETSON_GUIDE.md](docs/JETSON_GUIDE.md)** for recording, footage
folders, settings and troubleshooting.

---

## Settings

Everything adjustable is in **`config.yaml`**:

- **Camera:** device and resolution.
- **Recording:** frame rates, bitrates, and how many business days of footage to keep (7).
- **Liveness:** which actions can be asked for, how many per attempt, and the time allowed.
- **Face check:** the accept threshold (0.50), the reject threshold (0.25) and the quality limits.

## Privacy

- **Face data stays on the device** (`faces_db/`) and is **never** uploaded. The `.gitignore`
  keeps `faces_db/`, `users.db` (card IDs), personal greetings, snapshots and logs out of git.
- **Consent:** get each person's written consent before registering their face. Face data is
  biometric personal data under India's DPDP Act 2023.
- **Removal:** the Delete tab removes everything stored about a person.

## Project status

| Done | Still to do |
|---|---|
| Admin app, card registry, guided face registration, ArcFace verification, random liveness challenge, voice prompts, full-screen live recognition, DoorCam v2 recording | Connecting the RFID reader and the Arduino/servo to the door program (`main.py`), and a trial on the real Jetson hardware |

Tested on macOS (Python 3.11) and in an Ubuntu 18.04 / Python 3.6 / Tk 8.6 container that
matches the Jetson's software. The Jetson hardware itself (camera, GPU encoder, speaker) is
still to be tested on the device.
