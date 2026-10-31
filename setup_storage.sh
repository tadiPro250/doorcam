#!/bin/bash
# Mounts the 32 GB card (in a USB reader) at /footage on every boot.
#
#   bash setup_storage.sh                     list drives so you can find the card (e.g. /dev/sda1)
#   bash setup_storage.sh /dev/sda1 --format  ERASE the card and format it ext4 (recommended:
#                                             survives power cuts better than FAT/exFAT)
#   bash setup_storage.sh /dev/sda1           use the card as it is (ext4, vfat or exfat)
set -e
MNT=/footage
DEV="$1"

if [ -z "$DEV" ]; then
    lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT,TRAN
    echo
    echo "Find the 32 GB card above (TRAN=usb), then run:  bash setup_storage.sh /dev/sdX1 --format"
    exit 0
fi
[ -b "$DEV" ] || { echo "$DEV is not a drive/partition"; exit 1; }

BOOT_DISK=$(lsblk -no PKNAME "$(findmnt -n -o SOURCE /)")
case "$DEV" in
    /dev/"$BOOT_DISK"*) echo "Refusing: $DEV is on the boot card."; exit 1 ;;
esac

if [ "$2" = "--format" ]; then
    lsblk "$DEV"
    read -r -p "This ERASES everything on $DEV. Type ERASE to continue: " answer
    [ "$answer" = "ERASE" ] || { echo "Cancelled."; exit 1; }
    sudo umount "$DEV" 2>/dev/null || true
    sudo mkfs.ext4 -F -L FOOTAGE "$DEV"
fi

UUID=$(sudo blkid -s UUID -o value "$DEV")
FSTYPE=$(sudo blkid -s TYPE -o value "$DEV")
# nofail: the Jetson still boots (and DoorCam shows "NO STORAGE") if the card is missing.
case "$FSTYPE" in
    ext4)  OPTS="defaults,noatime,nofail,x-systemd.device-timeout=10"; PASS=2 ;;
    vfat)  OPTS="defaults,noatime,nofail,x-systemd.device-timeout=10,uid=$(id -u),gid=$(id -g),umask=022"; PASS=0 ;;
    exfat) sudo apt-get install -y exfat-fuse exfat-utils
           OPTS="defaults,noatime,nofail,x-systemd.device-timeout=10,uid=$(id -u),gid=$(id -g),umask=022"; PASS=0 ;;
    *)     echo "Unsupported filesystem '$FSTYPE'. Re-run with --format."; exit 1 ;;
esac

sudo umount "$DEV" 2>/dev/null || true   # the desktop may have auto-mounted it under /media
sudo mkdir -p "$MNT"
sudo sed -i "\#[[:space:]]$MNT[[:space:]]#d" /etc/fstab
echo "UUID=$UUID $MNT $FSTYPE $OPTS 0 $PASS" | sudo tee -a /etc/fstab
sudo mount "$MNT"
[ "$FSTYPE" = "ext4" ] && sudo chown "$USER:$USER" "$MNT"
df -h "$MNT"
echo "Footage card ready at $MNT"
