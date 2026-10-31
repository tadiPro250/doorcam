#!/usr/bin/env python3
"""Create the spoken prompt files in prompts/ with the Piper voice model.

    make_prompts.py --voice en_US-lessac-medium.onnx            # all standard prompts
    make_prompts.py --voice en_US-lessac-medium.onnx --name Ravi  # greeting for one person
    make_prompts.py --voice ... --force                          # remake existing files

Uses the piper-tts Python package if it is installed (laptop, Python 3.9+), otherwise the
standalone `piper` program (the Jetson: github.com/rhasspy/piper releases, linux_aarch64).
"""
import argparse
import os
import subprocess
import sys
import wave

import prompts

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def synth_package(voice_path):
    from piper import PiperVoice
    voice = PiperVoice.load(voice_path)

    def synth(text, out):
        with wave.open(out, "wb") as wf:
            voice.synthesize_wav(text, wf)
    return synth


def synth_binary(voice_path, exe):
    def synth(text, out):
        subprocess.run([exe, "--model", voice_path, "--output_file", out],
                       input=text.encode("utf-8"), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return synth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", required=True, help="Piper voice model (.onnx, with its .onnx.json next to it)")
    ap.add_argument("--out", default=os.path.join(APP_DIR, "prompts"))
    ap.add_argument("--name", action="append", default=[], help="make a 'Hello <name>' greeting")
    ap.add_argument("--piper", default="piper", help="standalone piper program (if the package isn't installed)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    try:
        synth = synth_package(args.voice)
    except ImportError:
        synth = synth_binary(args.voice, args.piper)
    os.makedirs(args.out, exist_ok=True)

    if args.name:  # only the greetings when names are given
        jobs = dict((prompts.hello_key(n), prompts.hello_text(n)) for n in args.name)
    else:
        jobs = dict(prompts.TEXTS)
    for key, text in sorted(jobs.items()):
        out = os.path.join(args.out, key + ".wav")
        if os.path.exists(out) and not args.force:
            continue
        synth(text, out)
        print("%-16s %s" % (key, text))


if __name__ == "__main__":
    sys.exit(main())
