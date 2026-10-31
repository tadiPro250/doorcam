# DoorCam v2

Live door feed on the monitor with person detection, face recognition (name boxes),
all-day recording and person event clips. For the original Jetson Nano (JetPack 4.6).

## Install (once)

1. Download the project to the Jetson's home folder:
   `git clone https://github.com/tadiPro250/doorcam.git ~/doorcam`
2. `cd ~/doorcam && bash setup.sh` — takes about 1.5 h, needs internet. It installs everything,
   sets the clock up (Asia/Kolkata, auto sync) and replaces the old `cam.sh` autostart.
3. Plug the 32 GB card in via a USB reader, then:
   `bash setup_storage.sh` (lists drives) → `bash setup_storage.sh /dev/sda1 --format`
4. Enroll people (see below), then reboot.

Full step-by-step guide for starting on power-on: **[START_ON_BOOT.md](START_ON_BOOT.md)**.
For the card + face check pieces (models, ONNX Runtime, Piper), see the main [README](../README.md#part-b-set-it-up-on-the-jetson-nano).

## Enrolling faces

```
python3 enroll.py Alice photo1.jpg photo2.jpg     # 3-5 clear front-facing photos
python3 enroll.py Alice --camera                  # quit DoorCam first (Q on the monitor)
python3 enroll.py Bob /footage/unknown_faces/2026-10-30_Fri/101530_t12.jpg
python3 enroll.py --list
python3 enroll.py --remove Alice
```
Unrecognised faces are saved to `/footage/unknown_faces/<day>/`; enroll any of them by name.
The running app picks up changes within 30 s.

## Door cards: the admin app

```
./install_admin_app.sh        # once: adds "DoorCam Admin" to the app menu and desktop
python3 admin_app.py          # or start it by hand (--fullscreen for the inside monitor)
```
Menu: **Cards**, **Assign**, **Revoke**, **Restore**, **Delete** and **Camera feed**.

- **Assign**: tap a card. A *new* card goes to a new person (type the name; their face is then
  registered at the camera with voice guidance, ~30 s) or to someone already registered. A card
  that is *already in use* shows its owner, so you can change their access level, re-register
  their face, revoke/restore it or delete the person. Each person has exactly one card: giving
  someone a new card replaces their old one.
- **Delete**: removes a person's card, face data and greeting, and shows how much storage
  each person uses.
- **Camera feed**: opens full screen with live face recognition (green = registered, red =
  unknown, yellow = person when the Jetson detector is available). Esc or Q closes it.
  Quit DoorCam first (Q on its monitor) so the camera is free.

Once an admin card exists, the app asks for one at start and locks itself after 5 minutes
unused. The same card actions work from a terminal with
`users.py list | assign | revoke | restore | delete`.

## Where things go

```
/footage/continuous/2026-10-30_Fri/09-00-00.mkv     all day, one file per hour, 640x480 @ 10 fps
/footage/events/2026-10-30_Fri/09-14-22_Alice.mkv   person clips with name boxes (+ .jpg snapshot)
/footage/unknown_faces/2026-10-30_Fri/...           faces it didn't recognise
/footage/events.db                                  event log (SQLite)
/footage/<kind>/unsynced/                           footage recorded before the clock synced
```
- The newest **7 weekdays** are kept; weekend footage is kept with its week and deleted with it.
- If free space drops below 1 GB the oldest files go first; event clips are capped at 3 GB.
- Play `.mkv` files with VLC. They stay playable even if the power was cut mid-recording.
- List today's visitors:
  `sqlite3 /footage/events.db "select start, names, clip from events order by id desc limit 20"`

On the monitor: **green** = recognised, **red** = face seen but unknown, **yellow** = person,
face not seen yet. `REC` shows while an event clip is recording. Q or ESC quits.

## Settings

Everything is in `config.yaml` (bitrates, how strict face matching is, clip lengths,
retention). Restart DoorCam after changing it (Q, then log out and in, or reboot).

## Troubleshooting

| Problem | Look at |
|---|---|
| App errors | `~/doorcam/logs/doorcam.log`, `~/doorcam/logs/console.log` |
| "CLOCK NOT SYNCED" on screen | `journalctl -u doorcam-timefix`, `chronyc tracking` |
| "NO STORAGE" on screen | `lsblk`, `df -h /footage` — is the card in? |
| Wrong names | add more photos of that person, or lower `faces.tolerance` (e.g. 0.45) |
| Faces never found | people too far away: raise `faces.upsample` to 2 |
| Run by hand to see errors | `cd ~/doorcam && python3 main.py` |
