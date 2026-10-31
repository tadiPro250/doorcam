# Making DoorCam start when the Jetson is powered on

Goal: power on the Jetson, and with nobody touching it the monitor shows the DoorCam
feed and recording starts.

How it works:

```
power on → Ubuntu boots → logs in automatically → desktop autostart
         → run_doorcam.sh → main.py (restarted automatically if it crashes)
```

`setup.sh` installs this for you. Below are the steps, plus how to check each part and
how to do it by hand.

---

## Step 1: get the project onto the Jetson

Open a terminal (Ctrl+Alt+T) and download it from GitHub:

```
sudo apt update && sudo apt install -y git
git clone https://github.com/tadiPro250/doorcam.git ~/doorcam
cd ~/doorcam
```

(Without internet: copy the `doorcam` folder from a pendrive instead, e.g.
`cp -r /media/$USER/<drive name>/doorcam ~/`; `ls /media/$USER` shows the drive name.)

## Step 2: run the setup (once, about 1.5 hours, needs internet)

```
bash setup.sh
```

Among other things, this:
- creates `~/.config/autostart/doorcam.desktop`, which starts DoorCam after login
- turns off the old `cam.sh` autostart so the two don't fight over the camera
- sets up automatic clock sync (Asia/Kolkata)

If it prints `!! cam.sh is also started from these files`, open each listed file and
remove or comment out (`#`) the line that runs `cam.sh`.

## Step 3: set up the 32 GB card for footage

Plug the card in with a USB reader, then:

```
bash setup_storage.sh                      # find the card in the list, e.g. /dev/sda1
bash setup_storage.sh /dev/sda1 --format   # ERASES the card; type ERASE to confirm
```

From now on the card is mounted at `/footage` at every boot.

## Step 4: make sure Ubuntu logs in automatically

The desktop autostart only runs once a user is logged in. v1 already started on boot,
so this is probably already on. To check:

```
grep -i AutomaticLogin /etc/gdm3/custom.conf
```

You should see these two lines, without a `#` in front:

```
AutomaticLoginEnable=true
AutomaticLogin=<your username>
```

If they're missing, run `sudo nano /etc/gdm3/custom.conf` and add them under `[daemon]`
(use your own username, from `whoami`). Save with Ctrl+O, then Enter, then exit with Ctrl+X.

You can also turn this on in Settings → Users → Unlock → Automatic Login: ON.

## Step 5: test it

```
sudo reboot
```

About 30-60 seconds after the desktop appears, DoorCam should open fullscreen.
The first start after setup takes longer, because the detection model loads.

Check that everything works:

| Check | Command / what to look for |
|---|---|
| DoorCam is running | `pgrep -af main.py` |
| App log | `tail -f ~/doorcam/logs/doorcam.log` (Ctrl+C to stop) |
| Using the hardware encoder | the log says `video encoder: nvv4l2h264enc` |
| Clock synced | `journalctl -u doorcam-timefix` says `clock synchronised`; the screen has no "CLOCK NOT SYNCED" |
| Footage card mounted | `df -h /footage` shows about 29 GB; the screen has no "NO STORAGE" |
| Recording | `ls /footage/continuous/*/` shows a file that keeps growing |

---

## Everyday use

- **Quit DoorCam:** press **Q** or **ESC** on the monitor. It stays closed until the next
  reboot or login.
- **Start it again without rebooting:** `bash ~/doorcam/run_doorcam.sh &`
- **Crashes:** DoorCam restarts itself after 5 seconds. Only one copy can run at a time.

## Turning autostart off

```
mv ~/.config/autostart/doorcam.desktop ~/.config/autostart/doorcam.desktop.off
```

Rename it back to `doorcam.desktop` to turn autostart on again.

## Setting up autostart by hand (only if you skipped setup.sh)

Create `~/.config/autostart/doorcam.desktop` with this content (replace `<user>`):

```
[Desktop Entry]
Type=Application
Name=DoorCam
Exec=/bin/bash /home/<user>/doorcam/run_doorcam.sh
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=5
```

Then disable the old launcher. Check each of these for anything that runs `cam.sh`:
- `~/.config/autostart/` (rename the old `.desktop` file to `.disabled`)
- `crontab -e` (put `#` in front of the `@reboot ... cam.sh` line)

## If it doesn't start after a reboot

1. **Is auto-login on?** If the login screen shows, see Step 4.
2. **Did it start and crash?** Look at the end of `~/doorcam/logs/console.log`.
3. **Does it run by hand?** `cd ~/doorcam && python3 main.py` shows errors directly.
4. **Is the old cam.sh still holding the camera?** Run `pgrep -af cam.sh`. If it shows
   anything, cam.sh is still being started from somewhere; see Step 2.
