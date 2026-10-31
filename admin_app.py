#!/usr/bin/env python3
"""DoorCam admin app for the inside monitor: list, assign, revoke and restore cards, camera view.

    python3 admin_app.py [--fullscreen]   (install_admin_app.sh adds it to the Jetson's app menu)

Once any admin card exists, the app opens only after an admin card is tapped, and locks
itself again after 5 minutes without use. Card fields accept a tap on a USB reader that types
the card ID, or the ID typed by hand. Keys: Ctrl+1..5 switch screens, Ctrl+L locks.
"""
import argparse
import glob
import os
import shutil
import sys
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import cv2
import yaml

import detector
import facepoints
import recognizer
import registration
import verify
from prompts import Speaker
from users import Users, normalise_uid

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOCK_AFTER_S = 300
TOAST_S = 3.5

# Semantic colour tokens (dark slate console; green = OK/active, red = destructive/revoked).
C = {
    "bg": "#0F172A", "sidebar": "#0B1220", "surface": "#1B2336", "surface_hi": "#243049",
    "row_alt": "#18202F", "border": "#334155", "border_hi": "#475569",
    "fg": "#F8FAFC", "muted": "#94A3B8", "disabled_fg": "#64748B",
    "accent": "#22C55E", "accent_hi": "#16A34A", "on_accent": "#0F172A",
    "danger": "#EF4444", "danger_hi": "#DC2626", "on_danger": "#FFFFFF",
    "warn": "#F59E0B", "focus": "#38BDF8", "select": "#334155",
}

NAV = [("list", "Cards", "list"), ("assign", "Assign", "card_plus"), ("revoke", "Revoke", "block"),
       ("restore", "Restore", "restore"), ("delete", "Delete", "trash"), ("camera", "Camera feed", "camera")]


def ppm(frame_bgr):
    """A frame as PPM bytes, which Tk shows without needing Pillow."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return b"P6 %d %d 255 " % (w, h) + rgb.tobytes()


def draw_icon(cv, name, color, size=20, width=2):
    """One consistent line-icon set, drawn so it stays sharp at any size."""
    cv.delete("all")
    s = size / 20.0
    k = dict(fill=color, width=width, capstyle="round", joinstyle="round")

    def p(*xy):
        return [v * s for v in xy]
    if name == "list":
        for y in (5, 10, 15):
            cv.create_oval(*p(2.5, y - 1, 4.5, y + 1), fill=color, outline=color)
            cv.create_line(*p(7, y, 18, y), **k)
    elif name == "card_plus":
        cv.create_rectangle(*p(2, 4, 15, 14), outline=color, width=width)
        cv.create_line(*p(2, 7.5, 15, 7.5), **k)
        cv.create_line(*p(16.5, 12, 16.5, 19), **k)
        cv.create_line(*p(13, 15.5, 20, 15.5), **k)
    elif name == "block":
        cv.create_oval(*p(3, 3, 17, 17), outline=color, width=width)
        cv.create_line(*p(5.5, 14.5, 14.5, 5.5), **k)
    elif name == "restore":
        cv.create_arc(*p(3, 3, 17, 17), start=110, extent=290, style="arc", outline=color, width=width)
        cv.create_line(*p(3.5, 3, 4.5, 7.5, 9, 6.5), **k)
    elif name == "camera":
        cv.create_rectangle(*p(2, 6, 18, 16), outline=color, width=width)
        cv.create_line(*p(7, 6, 8.5, 3.5, 11.5, 3.5, 13, 6), **k)
        cv.create_oval(*p(7, 8, 13, 14), outline=color, width=width)
    elif name == "camera_off":
        draw_icon(cv, "camera", color, size, width)
        cv.create_line(*p(2, 2, 18, 18), **k)
    elif name == "lock":
        cv.create_arc(*p(5, 2, 15, 12), start=0, extent=180, style="arc", outline=color, width=width)
        cv.create_line(*p(5, 7, 5, 9), **k)
        cv.create_line(*p(15, 7, 15, 9), **k)
        cv.create_rectangle(*p(3, 9, 17, 18), outline=color, width=width)
        cv.create_line(*p(10, 12.5, 10, 14.5), **k)
    elif name == "shield":
        cv.create_line(*p(10, 2, 17, 5, 16.5, 11, 10, 18, 3.5, 11, 3, 5, 10, 2), **k)
        cv.create_line(*p(6.8, 10, 9, 12.3, 13.4, 7.6), **k)
    elif name == "trash":
        cv.create_line(*p(3, 5.5, 17, 5.5), **k)
        cv.create_line(*p(8, 5.5, 8, 3, 12, 3, 12, 5.5), **k)
        cv.create_line(*p(5, 5.5, 6, 18, 14, 18, 15, 5.5), **k)
        cv.create_line(*p(8.5, 9, 8.5, 14.5), **k)
        cv.create_line(*p(11.5, 9, 11.5, 14.5), **k)
    elif name == "person":
        cv.create_oval(*p(6.5, 3, 13.5, 10), outline=color, width=width)
        cv.create_arc(*p(3, 11, 17, 25), start=0, extent=180, style="arc", outline=color, width=width)


def icon(parent, name, color, bg, size=20, width=2):
    cv = tk.Canvas(parent, width=size, height=size, bg=bg, highlightthickness=0, bd=0)
    draw_icon(cv, name, color, size, width)
    return cv


class Btn(tk.Label):
    """Flat button with hover, pressed, focus and disabled states (tk.Button ignores colours on macOS)."""
    KINDS = {"primary": ("accent", "accent_hi", "on_accent"), "danger": ("danger", "danger_hi", "on_danger"),
             "ghost": ("surface_hi", "border_hi", "fg")}

    def __init__(self, parent, text, command, kind="primary", font=None):
        self.bg_, self.hover_, self.fg_ = [C[t] for t in self.KINDS[kind]]
        super().__init__(parent, text=text, bg=self.bg_, fg=self.fg_, font=font, padx=20, pady=11,
                         cursor="hand2", takefocus=1, highlightthickness=2,
                         highlightbackground=parent.cget("bg"), highlightcolor=C["focus"])
        self.command, self.enabled = command, True
        self.bind("<Enter>", lambda _e: self._paint(self.hover_))
        self.bind("<Leave>", lambda _e: self._paint(self.bg_))
        self.bind("<ButtonPress-1>", lambda _e: self._paint(self.hover_))
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Return>", lambda _e: self.invoke())
        self.bind("<space>", lambda _e: self.invoke())

    def _paint(self, bg):
        if self.enabled:
            self.configure(bg=bg)

    def _release(self, event):
        self._paint(self.hover_)
        if 0 <= event.x <= self.winfo_width() and 0 <= event.y <= self.winfo_height():
            self.invoke()

    def set_enabled(self, on):
        self.enabled = on
        self.configure(bg=self.bg_ if on else C["border"], fg=self.fg_ if on else C["disabled_fg"],
                       cursor="hand2" if on else "arrow")

    def invoke(self):
        if self.enabled:
            self.command()


def folder_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%d %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1024.0


def fit(frame, w, h):
    """Scale a frame to fit inside w x h, keeping its shape."""
    scale = min(w / float(frame.shape[1]), h / float(frame.shape[0]))
    if abs(scale - 1.0) < 0.02:
        return frame, 1.0
    size = (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR), scale


def video_area(parent):
    """A black area for video. The picture is placed inside it, so it can never resize the area
    (packing the picture made the window grow with every frame)."""
    area = tk.Frame(parent, bg="black")
    label = tk.Label(area, bg="black", bd=0)
    label.place(relx=0, rely=0, relwidth=1, relheight=1)
    return area, label


class LiveView(tk.Toplevel):
    """Full-screen camera with live face recognition, in the style of the DoorCam monitor.
    Esc or Q closes it."""
    GREEN, RED, YELLOW = (80, 200, 80), (60, 60, 230), (0, 200, 255)

    def __init__(self, app, cap, rec, note=""):
        super().__init__(app, bg="black")
        self.app, self.cap, self.rec, self.note = app, cap, rec, note
        self.title("DoorCam - camera")
        self.geometry("%dx%d+0+0" % (self.winfo_screenwidth(), self.winfo_screenheight()))
        self.attributes("-fullscreen", True)
        self.label = tk.Label(self, bg="black", bd=0, fg="white", font=app.F["h2"])
        self.label.place(relx=0, rely=0, relwidth=1, relheight=1)
        for k in ("<Escape>", "<KeyPress-q>", "<KeyPress-Q>"):
            self.bind(k, lambda _e: self.close())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._fps, self._last, self._fail, self._job, self._photo = 0.0, time.monotonic(), 0, None, None
        self.focus_force()
        self._job = self.after(30, self._tick)

    def _tick(self):
        self._job = None
        ok, frame = self.cap.read()
        if not ok:
            self._fail += 1
            if self._fail > 30:
                self.label.configure(image="", text="The camera stopped sending pictures.\n\nPress Esc to close.")
                return
            self._job = self.after(30, self._tick)
            return
        self._fail = 0
        now = time.monotonic()
        self._fps = 0.9 * self._fps + 0.1 / max(now - self._last, 1e-3)
        self._last = now
        if self.rec is not None:
            self.rec.submit(frame)
        w, h = max(self.winfo_width(), 320), max(self.winfo_height(), 240)
        img, scale = fit(frame, w, h)
        img = img.copy() if img is frame else img
        lw = max(2, int(round(2 * scale)))
        fs = 0.6 * max(1.0, scale)
        bar = max(28, int(28 * scale))
        if self.rec is not None:
            for x1, y1, x2, y2 in self.rec.persons:
                cv2.rectangle(img, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)),
                              self.YELLOW, lw)
            for (x1, y1, x2, y2), name, score in self.rec.faces:
                color = self.GREEN if name else self.RED
                p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
                cv2.rectangle(img, p1, p2, color, lw)
                text = name or "Unknown"
                if score is not None:
                    text += "  %.2f" % score
                (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
                ty = max(p1[1], bar + th + base + 8)  # keep the name below the status bar
                cv2.rectangle(img, (p1[0], ty - th - base - 8), (p1[0] + tw + 10, ty), color, -1)
                cv2.putText(img, text, (p1[0] + 5, ty - base - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2,
                            cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (img.shape[1], bar), (0, 0, 0), -1)
        status = time.strftime("%a %d %b %Y  %H:%M:%S") + "   %.0f fps" % self._fps
        if self.rec is not None:
            status += "   %d registered" % len(self.rec.people)
        cv2.putText(img, status, (10, int(bar * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * max(1.0, scale),
                    (255, 255, 255), 1, cv2.LINE_AA)
        hint = "ESC = close" + ("   |   " + self.note if self.note else "")
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * max(1.0, scale), 1)
        cv2.putText(img, hint, (img.shape[1] - tw - 12, int(bar * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * max(1.0, scale), self.YELLOW, 1, cv2.LINE_AA)
        self._photo = tk.PhotoImage(data=ppm(img))
        self.label.configure(image=self._photo, text="")
        self._job = self.after(10, self._tick)

    def close(self):
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        if self.rec is not None:
            self.rec.stop()
            self.rec = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        app = self.app
        self.destroy()
        app.on_live_closed()


class AdminApp(tk.Tk):
    def __init__(self, cfg, users, faces, fullscreen=False):
        super().__init__()
        self.cfg, self.users, self.faces = cfg, users, faces
        self.title("DoorCam Admin")
        self.geometry("1180x720")
        self.minsize(1000, 600)
        self.configure(bg=C["bg"])
        if fullscreen:
            self.attributes("-fullscreen", True)
        self._fonts()
        self._style()

        self.page = None
        self.cap = None
        self.unlocked_by = None
        self.last_message = ""
        self._camera_job = self._toast = self._toast_job = None
        self._thumbs = {}
        self._last_activity = time.monotonic()
        self._verifier = self._points = self._speaker = None
        self.reg = None
        self.live = None

        self.sidebar = tk.Frame(self, bg=C["sidebar"], width=240)
        self.sidebar.pack_propagate(False)
        self._build_sidebar()
        self.body = tk.Frame(self, bg=C["bg"])
        self.body.pack(side="right", fill="both", expand=True)
        self.content = tk.Frame(self.body, bg=C["bg"])
        self.content.pack(fill="both", expand=True, padx=32, pady=28)

        self.bind_all("<Key>", self._touch, add="+")
        self.bind_all("<Button>", self._touch, add="+")
        for i, (key, _, _) in enumerate(NAV):
            self.bind_all("<Control-Key-%d>" % (i + 1), lambda _e, k=key: self._shortcut(k))
        self.bind_all("<Control-l>", lambda _e: self.grab_current() is None and self.lock())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(5000, self._check_idle)
        if self.users.admins():
            self.lock()
        else:
            self.show("list")

    # ---- look ------------------------------------------------------------

    def _fonts(self):
        fams = set(tkfont.families(self))
        ui = next((f for f in ("Inter", "Ubuntu", "Cantarell", "Noto Sans", "DejaVu Sans", "Helvetica Neue",
                               "Helvetica") if f in fams), "TkDefaultFont")
        mono = next((f for f in ("JetBrains Mono", "Ubuntu Mono", "DejaVu Sans Mono", "Menlo", "Courier")
                     if f in fams), "TkFixedFont")
        self.F = {
            "brand": (ui, 17, "bold"), "h1": (ui, 22, "bold"), "h2": (ui, 14, "bold"), "body": (ui, 12),
            "small": (ui, 11), "label": (ui, 11, "bold"), "nav": (ui, 13), "nav_on": (ui, 13, "bold"),
            "btn": (ui, 12, "bold"), "kpi": (ui, 26, "bold"), "mono": (mono, 15), "mono_small": (mono, 12),
            "name": (ui, 18, "bold"),
        }

    def _style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("Treeview", background=C["surface"], fieldbackground=C["surface"], foreground=C["fg"],
                     bordercolor=C["border"], lightcolor=C["surface"], darkcolor=C["surface"],
                     rowheight=40, font=self.F["body"], borderwidth=0)
        st.map("Treeview", background=[("selected", C["select"])], foreground=[("selected", C["fg"])])
        st.configure("Treeview.Heading", background=C["bg"], foreground=C["muted"], font=self.F["label"],
                     bordercolor=C["border"], lightcolor=C["bg"], darkcolor=C["bg"], relief="flat", padding=(10, 8))
        st.map("Treeview.Heading", background=[("active", C["surface_hi"])])
        st.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        for name in ("TScrollbar", "Vertical.TScrollbar"):
            st.configure(name, background=C["surface_hi"], troughcolor=C["bg"], bordercolor=C["bg"],
                         arrowcolor=C["muted"], lightcolor=C["surface_hi"], darkcolor=C["surface_hi"], gripcount=0)
        st.map("Vertical.TScrollbar", background=[("active", C["border_hi"])])
        st.configure("Dark.TCombobox", fieldbackground=C["bg"], background=C["surface_hi"],
                     foreground=C["fg"], arrowcolor=C["fg"], bordercolor=C["border"], lightcolor=C["bg"],
                     darkcolor=C["bg"], padding=10, arrowsize=16)
        st.map("Dark.TCombobox", fieldbackground=[("readonly", C["bg"])], foreground=[("readonly", C["fg"])],
               selectbackground=[("readonly", C["bg"])], selectforeground=[("readonly", C["fg"])],
               bordercolor=[("focus", C["focus"])])
        self.option_add("*TCombobox*Listbox.background", C["surface"])
        self.option_add("*TCombobox*Listbox.foreground", C["fg"])
        self.option_add("*TCombobox*Listbox.selectBackground", C["select"])
        self.option_add("*TCombobox*Listbox.font", self.F["body"])

    # ---- building blocks -------------------------------------------------

    def _build_sidebar(self):
        sb = self.sidebar
        brand = tk.Frame(sb, bg=C["sidebar"])
        brand.pack(fill="x", padx=20, pady=(24, 28))
        icon(brand, "shield", C["accent"], C["sidebar"], 28, 2).pack(side="left")
        t = tk.Frame(brand, bg=C["sidebar"])
        t.pack(side="left", padx=10)
        tk.Label(t, text="DoorCam", font=self.F["brand"], bg=C["sidebar"], fg=C["fg"]).pack(anchor="w")
        tk.Label(t, text="Access admin", font=self.F["small"], bg=C["sidebar"], fg=C["muted"]).pack(anchor="w")

        self.nav = {}
        for i, (key, label, ic) in enumerate(NAV):
            row = tk.Frame(sb, bg=C["sidebar"], cursor="hand2", takefocus=1, highlightthickness=2,
                           highlightbackground=C["sidebar"], highlightcolor=C["focus"])
            row.pack(fill="x", padx=12, pady=2)
            bar = tk.Frame(row, bg=C["sidebar"], width=4)
            bar.pack(side="left", fill="y")
            cv = icon(row, ic, C["muted"], C["sidebar"])
            cv.pack(side="left", padx=(14, 12), pady=13)
            lbl = tk.Label(row, text=label, font=self.F["nav"], bg=C["sidebar"], fg=C["muted"], anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            hint = tk.Label(row, text="Ctrl+%d" % (i + 1), font=self.F["small"], bg=C["sidebar"], fg=C["sidebar"])
            hint.pack(side="right", padx=10)
            self.nav[key] = (row, bar, cv, lbl, hint, ic)
            for w in (row, cv, lbl, hint):
                w.bind("<Button-1>", lambda _e, k=key: self.show(k))
                w.bind("<Enter>", lambda _e, k=key: self._nav_paint(k, hover=True))
                w.bind("<Leave>", lambda _e, k=key: self._nav_paint(k))
            row.bind("<Return>", lambda _e, k=key: self.show(k))

        foot = tk.Frame(sb, bg=C["sidebar"])
        foot.pack(side="bottom", fill="x", padx=16, pady=16)
        self.session = tk.Label(foot, text="", font=self.F["small"], bg=C["sidebar"], fg=C["muted"],
                                anchor="w", justify="left", wraplength=200)
        self.session.pack(fill="x", pady=(0, 10))
        Btn(foot, "Lock  ·  Ctrl+L", self.lock, "ghost", self.F["btn"]).pack(fill="x", pady=(0, 6))
        Btn(foot, "Quit", self.close, "ghost", self.F["btn"]).pack(fill="x")

    def _nav_paint(self, key, hover=False):
        row, bar, cv, lbl, hint, ic = self.nav[key]
        active = key == self.page or (key == "assign" and self.page == "register")
        bg = C["surface"] if active or hover else C["sidebar"]
        fg = C["fg"] if active or hover else C["muted"]
        for w in (row, cv, lbl, hint):
            w.configure(bg=bg)
        lbl.configure(fg=fg, font=self.F["nav_on"] if active else self.F["nav"])
        hint.configure(fg=C["muted"] if hover else bg)
        bar.configure(bg=C["accent"] if active else bg)
        draw_icon(cv, ic, C["accent"] if active else fg)

    def card(self, parent, pad=20, **pack):
        """A surface with a 1px border; returns (outer, body)."""
        outer = tk.Frame(parent, bg=C["border"])
        inner = tk.Frame(outer, bg=C["surface"])
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        body = tk.Frame(inner, bg=C["surface"])
        body.pack(fill="both", expand=True, padx=pad, pady=pad)
        if pack:
            outer.pack(**pack)
        return outer, body

    def header(self, title, sub=None):
        h = tk.Frame(self.content, bg=C["bg"])
        h.pack(fill="x", pady=(0, 20))
        left = tk.Frame(h, bg=C["bg"])
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text=title, font=self.F["h1"], bg=C["bg"], fg=C["fg"], anchor="w").pack(fill="x")
        if sub:
            tk.Label(left, text=sub, font=self.F["body"], bg=C["bg"], fg=C["muted"], anchor="w",
                     justify="left", wraplength=640).pack(fill="x", pady=(4, 0))
        right = tk.Frame(h, bg=C["bg"])
        right.pack(side="right", anchor="n")
        return right

    def field_label(self, parent, text, step):
        row = tk.Frame(parent, bg=parent.cget("bg"))
        row.pack(fill="x", pady=(0, 8))
        tk.Label(row, text=str(step), font=self.F["label"], bg=C["surface_hi"], fg=C["fg"], width=2).pack(
            side="left", padx=(0, 10))
        tk.Label(row, text=text, font=self.F["label"], bg=parent.cget("bg"), fg=C["fg"]).pack(side="left")

    def helper(self, parent, text="", color=None):
        lbl = tk.Label(parent, text=text, font=self.F["small"], bg=parent.cget("bg"), fg=color or C["muted"],
                       anchor="w", justify="left", wraplength=440)
        lbl.pack(fill="x", pady=(6, 0))
        return lbl

    def entry(self, parent, mono=True, width=22):
        return tk.Entry(parent, font=self.F["mono"] if mono else self.F["body"], width=width, bg=C["bg"],
                        fg=C["fg"], insertbackground=C["fg"], relief="flat", highlightthickness=2,
                        highlightbackground=C["border"], highlightcolor=C["focus"])

    def badge(self, parent, text, color):
        return tk.Label(parent, text=text, font=self.F["label"], bg=parent.cget("bg"), fg=color, padx=10, pady=3,
                        highlightthickness=1, highlightbackground=color)

    def segmented(self, parent, options, on_pick, bg):
        """Row of toggle labels; returns a function that selects one."""
        frame = tk.Frame(parent, bg=bg)
        btns = {}

        def pick(value, notify=True):
            for k, b in btns.items():
                b.configure(bg=C["select"] if k == value else C["surface"], fg=C["fg"] if k == value else C["muted"])
            if notify:
                on_pick(value)
        for value, label in options:
            b = tk.Label(frame, text=label, font=self.F["btn"], bg=C["surface"], fg=C["muted"], padx=18, pady=10,
                         cursor="hand2", takefocus=1, highlightthickness=2, highlightbackground=bg,
                         highlightcolor=C["focus"])
            b.pack(side="left", padx=(0, 2))
            b.bind("<Button-1>", lambda _e, v=value: pick(v))
            b.bind("<Return>", lambda _e, v=value: pick(v))
            b.bind("<space>", lambda _e, v=value: pick(v))
            btns[value] = b
        return frame, pick

    def table(self, parent, columns, height=10):
        frame = tk.Frame(parent, bg=C["border"])
        inner = tk.Frame(frame, bg=C["surface"])
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        tree = ttk.Treeview(inner, columns=[c[0] for c in columns], show="headings", height=height,
                            selectmode="browse")
        for key, title, width, stretch in columns:
            tree.heading(key, text=title, anchor="w")
            tree.column(key, width=width, anchor="w", stretch=stretch)
        tree.tag_configure("alt", background=C["row_alt"])
        tree.tag_configure("revoked", foreground=C["muted"])
        scroll = ttk.Scrollbar(inner, orient="vertical", command=tree.yview, style="Vertical.TScrollbar")
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        return frame, tree

    def thumb(self, folder, size):
        """A registered face crop of the person, to see who a card belongs to."""
        key = (folder, size)
        if key not in self._thumbs:
            img = None
            files = sorted(glob.glob(os.path.join(self.faces.folder, folder, "*.jpg")))
            if files:
                im = cv2.imread(files[0])
                if im is not None:
                    img = tk.PhotoImage(data=ppm(cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)))
            self._thumbs[key] = img
        return self._thumbs[key]

    def avatar(self, parent, folder, size=120):
        img = self.thumb(folder, size) if folder else None
        if img is not None:
            return tk.Label(parent, image=img, bg=parent.cget("bg"), highlightthickness=1,
                            highlightbackground=C["border_hi"])
        holder = tk.Frame(parent, bg=C["bg"], width=size, height=size, highlightthickness=1,
                          highlightbackground=C["border_hi"])
        holder.pack_propagate(False)
        icon(holder, "person", C["border_hi"], C["bg"], size // 2, 2).place(relx=0.5, rely=0.5, anchor="center")
        return holder

    def toast(self, text, kind="ok"):
        """Short confirmation in the top-right corner; errors are shown next to their field instead."""
        self.last_message = text
        self._hide_toast()
        color = {"ok": C["accent"], "warn": C["warn"]}.get(kind, C["danger"])
        t = tk.Frame(self.body, bg=C["surface_hi"], highlightthickness=1, highlightbackground=C["border_hi"])
        tk.Frame(t, bg=color, width=4).pack(side="left", fill="y")
        tk.Label(t, text=text, font=self.F["body"], bg=C["surface_hi"], fg=C["fg"], padx=16, pady=12,
                 wraplength=420, justify="left").pack(side="left")
        t.place(relx=1.0, x=-24, y=20, anchor="ne")
        self._toast = t
        self._toast_job = self.after(int(TOAST_S * 1000), self._hide_toast)

    def _hide_toast(self):
        if self._toast is not None:
            self.after_cancel(self._toast_job)
            self._toast.destroy()
            self._toast = None

    def confirm(self, title, message, action, danger=True):
        """Themed yes/no dialog. Returns True if the action was confirmed."""
        dlg = tk.Toplevel(self, bg=C["border"])
        dlg.title(title)
        dlg.transient(self)
        dlg.resizable(False, False)
        box = tk.Frame(dlg, bg=C["surface"])
        box.pack(fill="both", expand=True, padx=1, pady=1)
        inner = tk.Frame(box, bg=C["surface"])
        inner.pack(fill="both", expand=True, padx=28, pady=24)
        tk.Label(inner, text=title, font=self.F["h2"], bg=C["surface"], fg=C["fg"], anchor="w").pack(fill="x")
        tk.Label(inner, text=message, font=self.F["body"], bg=C["surface"], fg=C["muted"], anchor="w",
                 justify="left", wraplength=420).pack(fill="x", pady=(8, 22))
        result = {"ok": False}

        def done(ok):
            result["ok"] = ok
            dlg.grab_release()
            dlg.destroy()
        row = tk.Frame(inner, bg=C["surface"])
        row.pack(fill="x")
        Btn(row, action, lambda: done(True), "danger" if danger else "primary", self.F["btn"]).pack(side="right")
        cancel = Btn(row, "Cancel", lambda: done(False), "ghost", self.F["btn"])
        cancel.pack(side="right", padx=(0, 10))
        dlg.bind("<Escape>", lambda _e: done(False))
        dlg.protocol("WM_DELETE_WINDOW", lambda: done(False))
        dlg.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - dlg.winfo_reqwidth()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - dlg.winfo_reqheight()) // 3
        dlg.geometry("+%d+%d" % (max(0, x), max(0, y)))
        dlg.grab_set()
        cancel.focus_set()  # the safe choice has focus, so a stray Enter never revokes
        self.wait_window(dlg)
        return result["ok"] and not self.locked()  # locking while it was open counts as Cancel

    # ---- navigation & session -------------------------------------------

    def _touch(self, _event=None):
        self._last_activity = time.monotonic()

    def _check_idle(self):
        if self.live is not None:
            self._touch()  # someone is watching the camera
        if self.page != "login" and self.users.admins() and time.monotonic() - self._last_activity > LOCK_AFTER_S:
            self.lock()
        self.after(5000, self._check_idle)

    def locked(self):
        """True when admin cards exist and nobody has tapped one in."""
        return self.unlocked_by is None and bool(self.users.admins())

    def _shortcut(self, page):
        if self.page != "login" and self.grab_current() is None:
            self.show(page)

    def clear(self):
        self.stop_camera()
        self.reg = None
        self._hide_toast()
        for w in self.content.winfo_children():
            w.destroy()

    def show(self, page, **kw):
        if page != "login" and self.locked():
            page, kw = "login", {}  # e.g. a dialog confirmed after the app auto-locked
        if page != "login" and not self.sidebar.winfo_ismapped():
            self.sidebar.pack(side="left", fill="y", before=self.body)
        self.clear()
        self.page = page
        if page != "login":
            for key in self.nav:
                self._nav_paint(key)
            if self.unlocked_by:
                self.session.configure(text="Unlocked by %s" % self.unlocked_by, fg=C["muted"])
            else:
                self.session.configure(text="No admin cards yet: this app is open to anyone. "
                                            "Assign a card as Admin to protect it.", fg=C["warn"])
        getattr(self, "page_" + page)(**kw)

    def lock(self):
        if not self.users.admins():
            self.toast("Assign an admin card first: without one the app can't be locked.", "warn")
            return
        self.unlocked_by = None
        for w in self.winfo_children():  # close the camera view and any open dialog (it counts as Cancel)
            if isinstance(w, LiveView):
                w.close()
            elif isinstance(w, tk.Toplevel):
                w.destroy()
        self.sidebar.pack_forget()
        self.show("login")

    def close(self):
        self.stop_camera()
        self.destroy()

    # ---- screens ---------------------------------------------------------

    def page_login(self):
        wrap = tk.Frame(self.content, bg=C["bg"])
        wrap.place(relx=0.5, rely=0.45, anchor="center")
        _, body = self.card(wrap, pad=44, fill="both")
        icon(body, "lock", C["accent"], C["surface"], 48, 3).pack(pady=(0, 18))
        tk.Label(body, text="Tap an admin card", font=self.F["h1"], bg=C["surface"], fg=C["fg"]).pack()
        tk.Label(body, text="Hold an admin card on the reader, or type its ID and press Enter.",
                 font=self.F["body"], bg=C["surface"], fg=C["muted"]).pack(pady=(6, 22))
        e = self.entry(body, width=20)
        e.configure(justify="center")
        e.pack(ipady=8)
        err = tk.Label(body, text="", font=self.F["small"], bg=C["surface"], fg=C["danger"])
        err.pack(pady=(10, 0))
        e.focus_set()

        def check(_e=None):
            user = self.users.lookup(e.get())
            e.delete(0, "end")
            if user and user.admin and user.active:
                self.unlocked_by = user.name
                self.show("list")
                self.toast("Welcome, %s." % user.name)
            else:
                self.last_message = "That card is not an active admin card."
                err.configure(text=self.last_message)
                e.configure(highlightbackground=C["danger"])
        e.bind("<Return>", check)
        self.login_entry = e

    def page_list(self):
        users = self.users.all()
        active = [u for u in users if u.active]
        right = self.header("Cards", "Everyone who can open the door with a card.")
        Btn(right, "+  Assign a card", lambda: self.show("assign"), "primary", self.F["btn"]).pack()

        kpis = tk.Frame(self.content, bg=C["bg"])
        kpis.pack(fill="x", pady=(0, 20))
        revoked = len(users) - len(active)
        stats = [("Cards", len(users), C["fg"]), ("Active", len(active), C["accent"]),
                 ("Revoked", revoked, C["danger"] if revoked else C["fg"]),
                 ("Admins", len([u for u in active if u.admin]), C["fg"])]
        for i, (label, value, color) in enumerate(stats):
            outer, body = self.card(kpis, pad=16)
            outer.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 12, 0))
            kpis.grid_columnconfigure(i, weight=1, uniform="kpi")
            tk.Label(body, text=label.upper(), font=self.F["label"], bg=C["surface"], fg=C["muted"]).pack(anchor="w")
            tk.Label(body, text=str(value), font=self.F["kpi"], bg=C["surface"], fg=color).pack(anchor="w")

        if not users:
            _, body = self.card(self.content, pad=40, fill="x")
            icon(body, "card_plus", C["muted"], C["surface"], 40, 2).pack()
            tk.Label(body, text="No cards yet", font=self.F["h2"], bg=C["surface"], fg=C["fg"]).pack(pady=(12, 4))
            tk.Label(body, text="Assign a card to a registered face to let that person in.",
                     font=self.F["body"], bg=C["surface"], fg=C["muted"]).pack(pady=(0, 16))
            Btn(body, "Assign a card", lambda: self.show("assign"), "primary", self.F["btn"]).pack()
        else:
            bar = tk.Frame(self.content, bg=C["bg"])
            bar.pack(fill="x", pady=(0, 12))
            tk.Label(bar, text="Search", font=self.F["label"], bg=C["bg"], fg=C["muted"]).pack(side="left", padx=(0, 10))
            search = self.entry(bar, mono=False, width=28)
            search.pack(side="left", ipady=6)
            state = {"filt": "all"}

            def refill(_e=None):
                q = search.get().strip().lower()
                tree.delete(*tree.get_children())
                rows = [u for u in users if (state["filt"] == "all" or u.active == (state["filt"] == "active"))
                        and (not q or q in u.name.lower() or q in u.card.lower())]
                for i, u in enumerate(rows):
                    tags = (["alt"] if i % 2 else []) + ([] if u.active else ["revoked"])
                    tree.insert("", "end", iid=u.card, tags=tags, values=(
                        u.card, u.name, "Admin" if u.admin else "User",
                        "●  Active" if u.active else "○  Revoked", u.added))

            def set_filter(f):
                state["filt"] = f
                refill()
            seg, pick = self.segmented(bar, [("all", "All"), ("active", "Active"), ("revoked", "Revoked")],
                                       set_filter, C["bg"])
            seg.pack(side="right")
            frame, tree = self.table(self.content, [("card", "Card ID", 180, False), ("name", "Name", 220, True),
                                                    ("role", "Role", 110, False), ("status", "Status", 120, False),
                                                    ("added", "Assigned", 190, False)], height=8)
            frame.pack(fill="both", expand=True)
            search.bind("<KeyRelease>", refill)
            pick("all")
            self.list_widgets = dict(tree=tree, search=search, pick=pick)

        with_card = set(u.folder for u in users)
        no_card = [f for f in self.faces.names() if f not in with_card]
        if no_card:
            note = tk.Frame(self.content, bg=C["bg"])
            note.pack(fill="x", pady=(14, 0))
            tk.Label(note, text="Registered without a card:", font=self.F["body"], bg=C["bg"],
                     fg=C["muted"]).pack(side="left")
            for f in no_card:
                chip = Btn(note, self.faces.display_name(f) + "   +", lambda f=f: self.show("assign", person=f),
                           "ghost", self.F["btn"])
                chip.configure(padx=14, pady=6)
                chip.pack(side="left", padx=(10, 0))

    def page_assign(self, person=None, card=None):
        self.header("Assign a card", "Tap a card. A new card goes to a new or registered person; a card that is "
                                     "already in use shows its owner so you can change or remove it.")
        cols = tk.Frame(self.content, bg=C["bg"])
        cols.pack(fill="both", expand=True)
        form_outer, form = self.card(cols, pad=28)
        form_outer.pack(side="left", fill="both", expand=True)
        prev_outer, prev = self.card(cols, pad=28)
        prev_outer.configure(width=310)
        prev_outer.pack_propagate(False)  # fixed width, so the form doesn't shift as the preview changes
        prev_outer.pack(side="left", fill="y", padx=(20, 0))
        tk.Label(prev, text="PREVIEW", font=self.F["label"], bg=C["surface"], fg=C["muted"]).pack(anchor="w")
        pv = tk.Frame(prev, bg=C["surface"], width=260)
        pv.pack(fill="both", expand=True, pady=(14, 0))

        self.field_label(form, "Tap the card", 1)
        row = tk.Frame(form, bg=C["surface"])
        row.pack(fill="x")
        card_e = self.entry(row)
        card_e.pack(side="left", ipady=8)
        card_state = tk.Label(row, text="", font=self.F["label"], bg=C["surface"], fg=C["muted"], padx=10, pady=3)
        card_state.pack(side="left", padx=(14, 0))
        CARD_HELP = "Hold the card on the reader: it types the ID. Or type it and press Enter."
        card_help = self.helper(form, CARD_HELP)
        dyn = tk.Frame(form, bg=C["surface"])
        dyn.pack(fill="both", expand=True)
        st = {"key": None, "mode": "existing" if person in self.faces.names() else "new", "level": "user"}
        w = {}  # widgets of the current card state

        def uid():
            return normalise_uid(card_e.get())

        def preview(folder, name, card, role, badges=(), note=None):
            for c in pv.winfo_children():
                c.destroy()
            self.avatar(pv, folder).pack(anchor="w")
            tk.Label(pv, text=name or "No person yet", font=self.F["name"], bg=C["surface"],
                     fg=C["fg"] if name else C["muted"], wraplength=250, justify="left").pack(anchor="w", pady=(14, 2))
            tk.Label(pv, text=card or "No card yet", font=self.F["mono_small"], bg=C["surface"],
                     fg=C["fg"] if card else C["muted"]).pack(anchor="w")
            b = tk.Frame(pv, bg=C["surface"])
            b.pack(anchor="w", pady=(12, 0))
            items = [("Admin" if role == "admin" else "User", C["accent"] if role == "admin" else C["muted"])]
            line = None
            for i, (text, color) in enumerate(items + list(badges)):  # two per row so they never get cut off
                if i % 2 == 0:
                    line = tk.Frame(b, bg=C["surface"])
                    line.pack(anchor="w", pady=(0, 8))
                self.badge(line, text, color).pack(side="left", padx=(0, 8))
            if note:
                tk.Label(pv, text=note[0], font=self.F["small"], bg=C["surface"], fg=note[1], wraplength=250,
                         justify="left").pack(anchor="w", pady=(14, 0))

        # -- a card nobody has yet --------------------------------------------
        def build_new():
            folders = self.faces.names()
            names = [self.faces.display_name(f) for f in folders]
            tk.Frame(dyn, bg=C["surface"], height=20).pack()
            self.field_label(dyn, "Who is it for?", 2)
            seg, pick_mode = self.segmented(dyn, [("new", "New person"), ("existing", "Registered person")],
                                            lambda v: set_mode(v), C["surface"])
            seg.pack(anchor="w", pady=(0, 10))
            box = tk.Frame(dyn, bg=C["surface"])
            box.pack(fill="x")
            name = self.entry(box, mono=False, width=30)
            name.configure(font=self.F["h2"])
            labels = []
            for f, n in zip(folders, names):
                other = self.users.card_of(f)
                labels.append("%s   (has card %s)" % (n, other.card) if other else n)
            combo = ttk.Combobox(box, values=labels, state="readonly", style="Dark.TCombobox", font=self.F["body"],
                                 width=34)
            person_help = self.helper(dyn)
            tk.Frame(dyn, bg=C["surface"], height=20).pack()
            self.field_label(dyn, "Access level", 3)
            lseg, pick_level = self.segmented(dyn, [("user", "User"), ("admin", "Admin")], lambda v: set_level(v),
                                              C["surface"])
            lseg.pack(anchor="w")
            level_help = self.helper(dyn)
            actions = tk.Frame(dyn, bg=C["surface"])
            actions.pack(anchor="w", pady=(24, 0))
            go = Btn(actions, "Register face & assign", lambda: submit(), "primary", self.F["btn"])
            go.pack(side="left")
            Btn(actions, "Clear", lambda: self.show("assign"), "ghost", self.F["btn"]).pack(side="left", padx=(10, 0))

            def chosen():
                """(folder or None, display name) of the person this card would go to."""
                if st["mode"] == "existing":
                    lbl = combo.get()
                    return (folders[labels.index(lbl)], names[labels.index(lbl)]) if lbl in labels else (None, "")
                n = " ".join(name.get().split())
                return (self.faces.folder_for(n) if n and self.faces.exists(n) else None), n

            def set_mode(v):
                st["mode"] = v
                for c in box.winfo_children():
                    c.pack_forget()
                if v == "new":
                    name.pack(anchor="w", ipady=6)
                    name.focus_set()
                else:
                    combo.pack(anchor="w")
                refresh()

            def set_level(v):
                st["level"] = v
                level_help.configure(text="Admins can also open this app and manage cards." if v == "admin"
                                     else "Opens the door only.")
                refresh()

            def refresh(_e=None):
                folder, n = chosen()
                note, badges, help_text, help_color = None, [], "", C["muted"]
                if st["mode"] == "existing":
                    help_text = ("Choose a registered person." if names else
                                 "Nobody is registered yet: choose New person.")
                    help_color = C["muted"] if names else C["warn"]
                    go.configure(text="Assign card")
                elif folder:
                    help_text = "%s is already registered: this card will be given to them." % self.faces.display_name(folder)
                    help_color = C["warn"]
                    go.configure(text="Assign card")
                else:
                    help_text = "Their face is registered at the camera next (about 30 seconds)."
                    go.configure(text="Register face & assign")
                    if n:
                        badges.append(("Face: register next", C["warn"]))
                if folder:
                    other = self.users.card_of(folder)
                    if other and other.card != uid():
                        note = ("%s already has card %s. Each person has one card, so that card will stop "
                                "working." % (self.faces.display_name(folder), other.card), C["warn"])
                person_help.configure(text=help_text, fg=help_color)
                name.configure(highlightbackground=C["border"])
                preview(folder, n, uid(), st["level"], [("New card", C["accent"])] + badges, note)

            def submit():
                folder, n = chosen()
                if st["mode"] == "existing" and folder is None:
                    person_help.configure(text="Choose who this card belongs to.", fg=C["danger"])
                    self.last_message = "Choose who this card belongs to."
                    return
                if st["mode"] == "new" and len(n) < 2:
                    person_help.configure(text="Type the person's name (at least 2 letters).", fg=C["danger"])
                    name.configure(highlightbackground=C["danger"])
                    self.last_message = "Type the person's name."
                    return
                card, admin = uid(), st["level"] == "admin"
                if folder is None:
                    self.show("register", card=card, name=n, admin=admin)
                    return
                other = self.users.card_of(folder)
                if other and other.card != card and not self.confirm(
                        "Replace %s's card?" % self.faces.display_name(folder),
                        "%s has card %s now. Each person has only one card, so card %s will stop working and "
                        "card %s will open the door instead." % (self.faces.display_name(folder), other.card,
                                                                 other.card, card), "Replace card", danger=True):
                    return
                self.assign_card(card, self.faces.display_name(folder), folder, admin)
                self.show("assign")
                self.toast(self.last_message)

            name.bind("<KeyRelease>", refresh)
            name.bind("<Return>", lambda _e: submit())
            combo.bind("<<ComboboxSelected>>", refresh)
            if person in folders:
                combo.set(labels[folders.index(person)])
            pick_level(st["level"])
            pick_mode(st["mode"])
            w.update(name=name, combo=combo, pick_mode=pick_mode, pick_level=pick_level, button=go,
                     person_help=person_help, refresh=refresh)

        # -- a card that already belongs to someone ------------------------------
        def build_existing(user):
            has_face = user.folder in self.faces.names()
            tk.Frame(dyn, bg=C["surface"], height=20).pack()
            self.field_label(dyn, "Access level", 2)
            lseg, pick_level = self.segmented(dyn, [("user", "User"), ("admin", "Admin")],
                                              lambda v: level_changed(v), C["surface"])
            lseg.pack(anchor="w")
            level_help = self.helper(dyn)
            save = Btn(dyn, "Save access level", lambda: save_level(), "primary", self.F["btn"])
            save.pack(anchor="w", pady=(12, 0))
            tk.Frame(dyn, bg=C["border"], height=1).pack(fill="x", pady=22)
            self.field_label(dyn, "Other actions", 3)
            acts = tk.Frame(dyn, bg=C["surface"])
            acts.pack(anchor="w")
            Btn(acts, "Re-register face", lambda: self.show("register", card=user.card, name=user.name,
                                                             admin=user.admin), "ghost", self.F["btn"]).pack(side="left")
            if user.active:
                Btn(acts, "Revoke card", lambda: self.revoke_card(user, then="assign"), "ghost",
                    self.F["btn"]).pack(side="left", padx=(10, 0))
            else:
                Btn(acts, "Restore card", lambda: self.restore_card(user, then="assign"), "ghost",
                    self.F["btn"]).pack(side="left", padx=(10, 0))
            Btn(acts, "Delete person", lambda: self.delete_person(user.folder, then="assign"), "danger",
                self.F["btn"]).pack(side="left", padx=(10, 0))

            def level_changed(v):
                st["level"] = v
                changed = (v == "admin") != user.admin
                save.set_enabled(changed)
                level_help.configure(text=("Admins can also open this app and manage cards." if v == "admin"
                                           else "Opens the door only.") + ("  (not saved yet)" if changed else ""))

            def save_level():
                admin = st["level"] == "admin"
                if not admin and user.admin and len(self.users.admins()) == 1 and not self.confirm(
                        "Remove the last admin?", "Without an admin card, anyone at this screen can manage "
                        "cards. Make %s a normal user anyway?" % user.name, "Make user", danger=True):
                    return
                if admin and not self.users.admins() and self.unlocked_by is None:
                    self.unlocked_by = user.name  # the first admin stays signed in
                self.users.set_admin(user.card, admin)
                self.show("assign", card=user.card)
                self.toast("%s is now %s." % (user.name, "an admin" if admin else "a user"))

            st["level"] = "admin" if user.admin else "user"
            pick_level(st["level"])
            badges = [("Active", C["accent"]) if user.active else ("Revoked", C["danger"])]
            note = None if has_face else ("No face data for %s: re-register their face, or the card can't open "
                                          "the door." % user.name, C["danger"])
            preview(user.folder if has_face else None, user.name, user.card, st["level"], badges, note)
            w.update(pick_level=pick_level, save=save)

        def on_card(_e=None):
            u = uid()
            user = self.users.lookup(u) if u else None
            key = ("existing", u) if user else ("new" if u else "empty")
            if u:
                card_e.configure(highlightbackground=C["border"])
                card_help.configure(text=CARD_HELP, fg=C["muted"])
            if key == st["key"]:
                if key == "new":
                    w["refresh"]()
                return
            st["key"] = key
            for c in dyn.winfo_children():
                c.destroy()
            w.clear()
            if user:
                card_state.configure(text="In use: %s" % user.name, fg=C["warn"], highlightthickness=1,
                                     highlightbackground=C["warn"])
                build_existing(user)
            elif u:
                card_state.configure(text="New card", fg=C["accent"], highlightthickness=1,
                                     highlightbackground=C["accent"])
                build_new()
            else:
                card_state.configure(text="", highlightthickness=0)
                tk.Label(dyn, text="Tap a card to start. If the card is already in use you will see whose it is.",
                         font=self.F["body"], bg=C["surface"], fg=C["muted"], wraplength=440,
                         justify="left").pack(anchor="w", pady=(24, 0))
                preview(None, "", "", "user")

        card_e.bind("<KeyRelease>", on_card)
        card_e.bind("<Return>", lambda _e: (on_card(), w.get("name") is not None and st["mode"] == "new"
                                            and w["name"].focus_set()))
        if card:
            card_e.insert(0, card)
        on_card()
        card_e.focus_set()
        self.assign_widgets = dict(card=card_e, state=card_state, card_help=card_help, w=w, st=st, on_card=on_card)

    def assign_card(self, uid, name, folder, is_admin):
        if self.locked():
            return self.show("login")
        first_admin = is_admin and not self.users.admins()
        self.users.assign(uid, name, folder, is_admin)
        if first_admin and self.unlocked_by is None:
            self.unlocked_by = name  # whoever set up the first admin card stays signed in
            self.session.configure(text="Unlocked by %s" % name, fg=C["muted"])
        self.toast("Card %s now opens the door for %s%s." % (uid, name, " (admin)" if is_admin else ""))

    def revoke_card(self, user, then="revoke"):
        if user.admin and len(self.users.admins()) == 1:
            msg = ("This is the last active admin card. Without it, anyone at this screen can manage "
                   "cards. %s's card will also stop opening the door." % user.name)
        else:
            msg = ("%s's card will stop opening the door at once. Their face stays registered, so the "
                   "card can be restored later." % user.name)
        if not self.confirm("Revoke card %s?" % user.card, msg, "Revoke card", danger=True):
            return
        self.users.set_active(user.card, False)
        self.show(then, **({"card": user.card} if then == "assign" else {}))
        self.toast("Card %s (%s) revoked." % (user.card, user.name))

    def restore_card(self, user, then="restore"):
        self.users.set_active(user.card, True)
        self.show(then, **({"card": user.card} if then == "assign" else {}))
        self.toast("Card %s (%s) opens the door again." % (user.card, user.name))

    def delete_person(self, folder, then="delete"):
        """Remove a person completely: card, face data and greeting."""
        card = self.users.card_of(folder)
        name = card.name if card else self.faces.display_name(folder)
        extra = ""
        if card and card.admin and card.active and len(self.users.admins()) == 1:
            extra = " This is the last admin card: afterwards anyone at this screen can manage cards."
        if not self.confirm("Delete %s?" % name, "This removes %s's card%s, face data and greeting from this "
                            "device. It can't be undone; to let them in again, register them from scratch.%s"
                            % (name, " (%s)" % card.card if card else "", extra), "Delete", danger=True):
            return
        self.users.delete_person(folder)
        self.faces.remove(folder)
        greeting = os.path.join(APP_DIR, "prompts", "hello_%s.wav" % folder)
        if os.path.exists(greeting):
            os.remove(greeting)
        self._thumbs = {}
        if card and self.unlocked_by == card.name and card.admin and not self.users.admins():
            self.unlocked_by = None
        self.show(then)
        self.toast("%s deleted." % name)

    # ---- registering a new face (in the app, at the door camera) ---------

    def _reg_tools(self):
        """Face model, landmarks and speaker, loaded on first use. Returns an error text or None."""
        try:
            if self._verifier is None:
                self._verifier = verify.Verifier(self.cfg.get("verify"), APP_DIR)
                self._verifier.db = self.faces
        except Exception as e:  # missing onnxruntime / model file
            return "The face model could not be loaded (%s). Is onnxruntime installed and models/ copied?" % e
        if self._points is None:
            self._points = facepoints.landmarker()
            if self._points is None:
                return "No face-landmark library found: install face_recognition (Jetson) or mediapipe."
        if self._speaker is None:
            self._speaker = Speaker(os.path.join(APP_DIR, "prompts"))
        return None

    def page_register(self, card, name, admin):
        self.header("Register %s's face" % name, "Stand about an arm's length from the camera in good light. "
                                                 "Follow the voice; it takes about 30 seconds.")
        cols = tk.Frame(self.content, bg=C["bg"])
        cols.pack(fill="both", expand=True)
        _, vbody = self.card(cols, pad=0, side="left", fill="both", expand=True)
        area, self.video = video_area(vbody)
        area.pack(fill="both", expand=True)
        side_outer, side = self.card(cols, pad=24)
        side_outer.configure(width=320)
        side_outer.pack_propagate(False)
        side_outer.pack(side="left", fill="y", padx=(20, 0))
        self.reg_ui = ui = {}
        ui["pose"] = tk.Label(side, text="", font=self.F["h2"], bg=C["surface"], fg=C["fg"], wraplength=260,
                              justify="left", anchor="w")
        ui["pose"].pack(fill="x")
        ui["msg"] = tk.Label(side, text="", font=self.F["body"], bg=C["surface"], fg=C["muted"], wraplength=260,
                             justify="left", anchor="w")
        ui["msg"].pack(anchor="w", pady=(6, 16))
        ui["dial"] = tk.Canvas(side, width=200, height=200, bg=C["surface"], highlightthickness=0)
        ui["dial"].pack(pady=(0, 16))
        tk.Label(side, text="PROGRESS", font=self.F["label"], bg=C["surface"], fg=C["muted"]).pack(anchor="w")
        ui["bar"] = tk.Canvas(side, width=260, height=10, bg=C["bg"], highlightthickness=0)
        ui["bar"].pack(anchor="w", pady=(6, 4))
        ui["count"] = tk.Label(side, text="", font=self.F["small"], bg=C["surface"], fg=C["muted"])
        ui["count"].pack(anchor="w")
        Btn(side, "Cancel", lambda: self.show("assign"), "ghost", self.F["btn"]).pack(fill="x", pady=(24, 0))

        error = self._reg_tools()
        if error is None:
            cam = self.cfg.get("camera", {})
            dev = cam.get("device", 0)
            self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(dev)
            if not self.cap.isOpened():
                self.stop_camera()
                error = ("The camera is busy or unplugged. If DoorCam is running it is using the camera: "
                         "press Q on its window, register the face here, then start DoorCam again.")
        if error:
            self._reg_message(error, retry=lambda: self.show("register", card=card, name=name, admin=admin))
            self.last_message = error
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.get("width", 640))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 480))
        self.reg = registration.Session(self._verifier, self._speaker)
        self.reg_target = (card, name, admin)
        self._reg_fail = 0
        self._camera_job = self.after(30, self._reg_step)

    def _reg_message(self, text, retry=None, ok=False):
        for w in self.video.winfo_children():
            w.destroy()
        self.video.configure(image="")
        box = tk.Frame(self.video, bg="black")
        box.place(relx=0.5, rely=0.5, anchor="center")
        icon(box, "shield" if ok else "camera_off", C["accent"] if ok else C["muted"], "black", 48, 2).pack()
        tk.Label(box, text=text, font=self.F["body"], bg="black", fg=C["fg"], wraplength=440,
                 justify="center").pack(pady=(14, 0))
        if retry:
            Btn(box, "Try again", retry, "ghost", self.F["btn"]).pack(pady=(18, 0))

    def _reg_step(self):
        session = self.reg
        if session is None or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self._reg_fail += 1
            if self._reg_fail > 30:
                self.stop_camera()
                card, name, admin = self.reg_target
                self._reg_message("The camera stopped sending pictures.",
                                  retry=lambda: self.show("register", card=card, name=name, admin=admin))
                return
            self._camera_job = self.after(30, self._reg_step)
            return
        self._reg_fail = 0
        session.feed(frame, self._points.faces(frame), time.monotonic())
        self._reg_draw(frame, session)
        if session.failed:
            self.stop_camera()
            card, name, admin = self.reg_target
            self._reg_message(session.failed + ".",
                              retry=lambda: self.show("register", card=card, name=name, admin=admin))
            self.last_message = session.failed
            return
        if session.finished:
            self.stop_camera()
            self._reg_message("Checking the pictures...", ok=True)
            self.after(50, self._reg_finish)
            return
        self._camera_job = self.after(10, self._reg_step)

    def _reg_draw(self, frame, session):
        ui, guide = self.reg_ui, session.guide
        view, _ = fit(cv2.flip(frame, 1), max(self.video.winfo_width(), 160),  # a mirror is easier to follow
                      max(self.video.winfo_height(), 120))
        self._photo = tk.PhotoImage(data=ppm(view))
        self.video.configure(image=self._photo)
        ui["pose"].configure(text=guide.pose[0] if not guide.finished else "Done")
        ui["msg"].configure(text=session.msg, fg=C["accent"] if session.in_zone or session.msg == "Good" else C["muted"])
        # dial: the dot is where the head points (mirrored), the box is where it should go
        d, r, cx, cy = ui["dial"], 90, 100, 100
        sc = r / 40.0
        d.delete("all")
        d.create_oval(cx - r, cy - r, cx + r, cy + r, fill=C["bg"], outline=C["border_hi"])
        d.create_line(cx - r, cy, cx + r, cy, fill=C["border"])
        d.create_line(cx, cy - r, cx, cy + r, fill=C["border"])
        if not guide.finished:
            _, axis, direction, _ = guide.pose
            lo, hi = registration.TARGET_DEG
            lv, sd = registration.LEVEL_DEG, registration.STRAIGHT_DEG
            if axis is None:
                box = (-sd, -sd, sd, sd)
            else:
                a, b = sorted((-direction * lo, -direction * hi))
                box = (a, -lv, b, lv) if axis == "yaw" else (-lv, a, lv, b)
            color = C["accent"] if session.in_zone else C["warn"]
            d.create_rectangle(cx + box[0] * sc, cy + box[1] * sc, cx + box[2] * sc, cy + box[3] * sc,
                               outline=color, width=2)
            rel = guide.relative()
            if rel is not None:
                dx = max(-r, min(r, -rel[0] * sc))
                dy = max(-r, min(r, -rel[1] * sc))
                d.create_oval(cx + dx - 8, cy + dy - 8, cx + dx + 8, cy + dy + 8, fill=color if session.in_zone
                              else C["fg"], outline="")
        frac = len(session.faces) / float(session.total)
        ui["bar"].delete("all")
        ui["bar"].create_rectangle(0, 0, 260 * frac, 10, fill=C["accent"], outline="")
        ui["count"].configure(text="%d of %d pictures" % (len(session.faces), session.total))

    def _reg_finish(self):
        session, (card, name, admin) = self.reg, self.reg_target
        if session is None or self.page != "register":
            return
        other, score = session.lookalike(name)
        if other and not self.confirm("This face looks like %s" % other,
                                      "It is very similar (%.2f) to %s, who is already registered. If this is "
                                      "the same person, cancel and give them the card as a registered person. "
                                      "Save it as %s anyway?" % (score, other, name), "Save anyway", danger=True):
            return self.show("assign")
        ok, message = session.save(name)
        if not ok:
            self.last_message = message
            return self._reg_message(message, retry=lambda: self.show("register", card=card, name=name, admin=admin))
        self._thumbs = {}
        folder = self.faces.folder_for(name)
        registration.make_greeting(name)
        self.reg = None
        self.assign_card(card, name, folder, admin)
        done = self.last_message
        self.show("list")
        self.toast(done)

    def _pick_page(self, title, sub, active, action_text, action, danger):
        self.header(title, sub)
        cols = tk.Frame(self.content, bg=C["bg"])
        cols.pack(fill="both", expand=True)
        left = tk.Frame(cols, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True)
        det_outer, det = self.card(cols, pad=28)
        det_outer.configure(width=310)
        det_outer.pack_propagate(False)
        det_outer.pack(side="left", fill="y", padx=(20, 0))

        bar = tk.Frame(left, bg=C["bg"])
        bar.pack(fill="x", pady=(0, 12))
        tk.Label(bar, text="Tap card", font=self.F["label"], bg=C["bg"], fg=C["muted"]).pack(side="left", padx=(0, 10))
        tap = self.entry(bar, width=20)
        tap.pack(side="left", ipady=6)
        tap_help = tk.Label(bar, text="", font=self.F["small"], bg=C["bg"], fg=C["danger"])
        tap_help.pack(side="left", padx=12)
        rows = [u for u in self.users.all() if u.active == active]
        frame, tree = self.table(left, [("card", "Card ID", 170, False), ("name", "Name", 200, True),
                                        ("role", "Role", 100, False), ("added", "Assigned", 180, False)], height=9)
        frame.pack(fill="both", expand=True)
        for i, u in enumerate(rows):
            tree.insert("", "end", iid=u.card, tags=["alt"] if i % 2 else [],
                        values=(u.card, u.name, "Admin" if u.admin else "User", u.added))

        def detail(_e=None):
            for w in det.winfo_children():
                w.destroy()
            sel = tree.selection()
            user = self.users.lookup(sel[0]) if sel else None
            if user is None:
                tk.Label(det, text="NO CARD SELECTED", font=self.F["label"], bg=C["surface"],
                         fg=C["muted"]).pack(anchor="w")
                tk.Label(det, text="Choose a card in the list,\nor tap it on the reader.", font=self.F["body"],
                         bg=C["surface"], fg=C["muted"], justify="left", width=24, anchor="w").pack(anchor="w", pady=(10, 0))
                return
            tk.Label(det, text="SELECTED CARD", font=self.F["label"], bg=C["surface"], fg=C["muted"]).pack(anchor="w")
            self.avatar(det, user.folder).pack(anchor="w", pady=(14, 0))
            tk.Label(det, text=user.name, font=self.F["name"], bg=C["surface"], fg=C["fg"]).pack(anchor="w", pady=(14, 2))
            tk.Label(det, text=user.card, font=self.F["mono_small"], bg=C["surface"], fg=C["fg"]).pack(anchor="w")
            row = tk.Frame(det, bg=C["surface"])
            row.pack(anchor="w", pady=(12, 0))
            self.badge(row, "Active" if user.active else "Revoked",
                       C["accent"] if user.active else C["danger"]).pack(side="left")
            self.badge(row, "Admin" if user.admin else "User", C["muted"]).pack(side="left", padx=(8, 0))
            tk.Label(det, text="Assigned %s" % user.added, font=self.F["small"], bg=C["surface"],
                     fg=C["muted"]).pack(anchor="w", pady=(12, 0))
            Btn(det, action_text, lambda: action(user), "danger" if danger else "primary",
                self.F["btn"]).pack(fill="x", pady=(26, 0))

        def find(_e=None):
            uid = normalise_uid(tap.get())
            tap.delete(0, "end")
            if tree.exists(uid):
                tree.selection_set(uid)
                tree.see(uid)
                tap_help.configure(text="")
            else:
                self.last_message = "Card %s isn't %s." % (uid, "active" if active else "revoked")
                tap_help.configure(text=self.last_message, fg=C["danger"])
        tap.bind("<Return>", find)
        tree.bind("<<TreeviewSelect>>", detail)
        detail()
        tap.focus_set()
        if not rows:
            tap_help.configure(text="No %s cards." % ("active" if active else "revoked"), fg=C["muted"])
        self.pick_widgets = dict(tree=tree, tap=tap, tap_help=tap_help, detail=det)

    def page_revoke(self):
        self._pick_page("Revoke a card", "The card stops opening the door immediately. Face data is kept.",
                        True, "Revoke card", self.revoke_card, True)

    def page_restore(self):
        self._pick_page("Restore a card", "Revoked cards. Restoring one lets it open the door again.",
                        False, "Restore card", self.restore_card, False)

    def page_delete(self):
        self.header("Delete", "Remove a person completely: their card, face data and greeting. "
                              "This frees storage and can't be undone.")
        folders = self.faces.names()
        rows = []
        for f in folders:
            u = self.users.card_of(f)
            rows.append((f, self.faces.display_name(f), u, folder_size(os.path.join(self.faces.folder, f))))
        known = set(folders)
        for u in self.users.all():  # cards whose face data is already gone
            if u.folder not in known:
                rows.append((u.folder, u.name, u, 0))
        snaps = os.path.join(APP_DIR, "snapshots")
        snap_size = folder_size(snaps)

        kpis = tk.Frame(self.content, bg=C["bg"])
        kpis.pack(fill="x", pady=(0, 20))
        try:
            free = shutil.disk_usage(APP_DIR).free
        except OSError:
            free = 0
        stats = [("People", str(len(folders))), ("Face data", human_size(sum(r[3] for r in rows))),
                 ("Snapshots", human_size(snap_size)), ("Free space", human_size(free))]
        for i, (label, value) in enumerate(stats):
            outer, body = self.card(kpis, pad=16)
            outer.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 12, 0))
            kpis.grid_columnconfigure(i, weight=1, uniform="kpi")
            tk.Label(body, text=label.upper(), font=self.F["label"], bg=C["surface"], fg=C["muted"]).pack(anchor="w")
            tk.Label(body, text=value, font=self.F["kpi"], bg=C["surface"], fg=C["fg"]).pack(anchor="w")

        cols = tk.Frame(self.content, bg=C["bg"])
        cols.pack(fill="both", expand=True)
        left = tk.Frame(cols, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True)
        det_outer, det = self.card(cols, pad=28)
        det_outer.configure(width=310)
        det_outer.pack_propagate(False)
        det_outer.pack(side="left", fill="y", padx=(20, 0))
        bar = tk.Frame(left, bg=C["bg"])
        bar.pack(fill="x", pady=(0, 12))
        tk.Label(bar, text="Tap card", font=self.F["label"], bg=C["bg"], fg=C["muted"]).pack(side="left", padx=(0, 10))
        tap = self.entry(bar, width=20)
        tap.pack(side="left", ipady=6)
        tap_help = tk.Label(bar, text="", font=self.F["small"], bg=C["bg"], fg=C["danger"])
        tap_help.pack(side="left", padx=12)
        if snap_size:
            Btn(bar, "Delete snapshots", lambda: delete_snaps(), "ghost", self.F["btn"]).pack(side="right")
        frame, tree = self.table(left, [("name", "Name", 200, True), ("card", "Card", 160, False),
                                        ("status", "Status", 110, False), ("size", "Storage", 100, False)], height=8)
        frame.pack(fill="both", expand=True)
        by_iid = {}
        for i, (f, name, u, size) in enumerate(rows):
            iid = "p%d" % i
            by_iid[iid] = (f, name, u, size)
            status = ("Revoked" if not u.active else "Active") if u else "No card"
            if f not in known:
                status = "No face data"
            tree.insert("", "end", iid=iid, tags=["alt"] if i % 2 else [],
                        values=(name, u.card if u else "-", status, human_size(size) if size else "-"))

        def detail(_e=None):
            for c in det.winfo_children():
                c.destroy()
            sel = tree.selection()
            if not sel:
                tk.Label(det, text="NOTHING SELECTED", font=self.F["label"], bg=C["surface"], fg=C["muted"]).pack(anchor="w")
                tk.Label(det, text="Choose a person in the list,\nor tap their card.", font=self.F["body"],
                         bg=C["surface"], fg=C["muted"], justify="left", width=24, anchor="w").pack(anchor="w", pady=(10, 0))
                return
            f, name, u, size = by_iid[sel[0]]
            tk.Label(det, text="SELECTED", font=self.F["label"], bg=C["surface"], fg=C["muted"]).pack(anchor="w")
            self.avatar(det, f if f in known else None).pack(anchor="w", pady=(14, 0))
            tk.Label(det, text=name, font=self.F["name"], bg=C["surface"], fg=C["fg"], wraplength=240,
                     justify="left").pack(anchor="w", pady=(14, 2))
            tk.Label(det, text=u.card if u else "No card", font=self.F["mono_small"], bg=C["surface"],
                     fg=C["fg"] if u else C["muted"]).pack(anchor="w")
            tk.Label(det, text="Uses %s" % human_size(size) if size else "No face data stored", font=self.F["small"],
                     bg=C["surface"], fg=C["muted"]).pack(anchor="w", pady=(12, 0))
            Btn(det, "Delete person", lambda: self.delete_person(f), "danger", self.F["btn"]).pack(fill="x", pady=(26, 0))

        def find(_e=None):
            uid = normalise_uid(tap.get())
            tap.delete(0, "end")
            for iid, (f, name, u, size) in by_iid.items():
                if u and u.card == uid:
                    tree.selection_set(iid)
                    tree.see(iid)
                    tap_help.configure(text="")
                    return
            self.last_message = "No one has card %s." % uid
            tap_help.configure(text=self.last_message)

        def delete_snaps():
            if not self.confirm("Delete all snapshots?", "Removes every saved camera snapshot (%s)."
                                % human_size(snap_size), "Delete", danger=True):
                return
            shutil.rmtree(snaps, ignore_errors=True)
            self.show("delete")
            self.toast("Snapshots deleted (%s freed)." % human_size(snap_size))

        tap.bind("<Return>", find)
        tree.bind("<<TreeviewSelect>>", detail)
        detail()
        tap.focus_set()
        if not rows:
            tap_help.configure(text="Nobody is registered.", fg=C["muted"])
        self.delete_widgets = dict(tree=tree, tap=tap, tap_help=tap_help, detail=det, rows=by_iid)

    def page_camera(self):
        self.header("Camera feed", "Full-screen live view with face recognition, like the DoorCam monitor. "
                                   "Press Esc or Q to come back here.")
        _, body = self.card(self.content, pad=40, fill="x")
        icon(body, "camera", C["accent"], C["surface"], 40, 2).pack()
        self.cam_status = tk.Label(body, text="", font=self.F["body"], bg=C["surface"], fg=C["muted"],
                                   wraplength=560, justify="center")
        self.cam_status.pack(pady=(14, 18))
        Btn(body, "Open full screen", self.open_live, "primary", self.F["btn"]).pack()
        self.open_live()

    def open_live(self):
        if self.live is not None:
            self.live.focus_force()
            return
        cam = self.cfg.get("camera", {})
        dev = cam.get("device", 0)
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(dev)
        if not cap.isOpened():
            cap.release()
            msg = ("The camera is busy or unplugged. If DoorCam is running it already shows this view on the "
                   "monitor; press Q there first to use it here.")
            self.last_message = msg
            if self.page == "camera":
                self.cam_status.configure(text=msg, fg=C["warn"])
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.get("width", 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 480))
        rec, note = None, ""
        error = self._reg_tools()
        if error is None:
            persons = None
            if detector._import_jetson()[0] is not None:  # GPU person detector on the Jetson
                try:
                    persons = detector.PersonDetector(self.cfg["detection"])
                except Exception as e:
                    note = "person detection off (%s)" % e
            points = facepoints.landmarker(max_faces=4)
            if points is None:
                note = "recognition off: no face library (install face_recognition or mediapipe)"
            else:
                rec = recognizer.Recognizer(self._verifier, points, self.faces, persons)
                rec.start()
        else:
            note = "recognition off: " + error
        self.live = LiveView(self, cap, rec, note)
        if self.page == "camera":
            self.cam_status.configure(text="The camera is open full screen. Press Esc or Q there to close it.",
                                      fg=C["muted"])

    def on_live_closed(self):
        self.live = None
        self._touch()
        if self.page == "camera":
            self.cam_status.configure(text="Closed. Open it again with the button below.", fg=C["muted"])

    def stop_camera(self):
        if self.live is not None:
            self.live.close()
        if self._camera_job is not None:
            self.after_cancel(self._camera_job)
            self._camera_job = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--db", default=os.path.join(APP_DIR, "users.db"))
    ap.add_argument("--faces", help="face folder (default: verify.db_dir in config.yaml)")
    args = ap.parse_args()
    with open(os.path.join(APP_DIR, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    faces_dir = args.faces or (cfg.get("verify") or {}).get("db_dir", verify.DEFAULTS["db_dir"])
    if not os.path.isabs(faces_dir):
        faces_dir = os.path.join(APP_DIR, faces_dir)
    AdminApp(cfg, Users(args.db), verify.FaceDB(faces_dir), args.fullscreen).mainloop()


if __name__ == "__main__":
    main()
