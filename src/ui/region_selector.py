import tkinter as tk
from typing import Callable, Optional


class RegionSelector(tk.Toplevel):
    """
    Tam ekran saydam overlay; fare surukleme ile (left, top, width, height) doner.
    Bagimsiz Toplevel olarak calisir, CustomTkinter veya Tkinter master'i ile uyumludur.
    """

    FILL_COLOR = "#1a3a5c"
    OUTLINE_COLOR = "#4da6ff"
    HINT_BG = "#1a1a1a"
    HINT_FG = "#ffffff"
    HINT_TEXT = "Altyazi bölgesini seçmek için sürükleyin   |   ESC = iptal"

    def __init__(self, master, callback: Callable[[Optional[tuple]], None]):
        super().__init__(master)
        self._callback = callback
        self._start_x = self._start_y = 0
        self._rect_id: Optional[int] = None
        self._coord_id: Optional[int] = None

        self.overrideredirect(True)
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{sw}x{sh}+0+0")
        self.attributes("-alpha", 0.30)
        self.configure(bg="black")
        self.lift()
        self.focus_force()

        self._canvas = tk.Canvas(
            self,
            cursor="crosshair",
            bg="black",
            highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True)

        self._canvas.create_rectangle(
            0, 0, sw, 40,
            fill=self.HINT_BG,
            outline="",
        )
        self._canvas.create_text(
            sw // 2, 20,
            text=self.HINT_TEXT,
            fill=self.HINT_FG,
            font=("Segoe UI", 13),
            anchor="center",
        )

        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Escape>", lambda _: self._cancel())

    def _on_press(self, event: tk.Event):
        self._start_x = event.x
        self._start_y = event.y
        self._clear_rect()

    def _on_drag(self, event: tk.Event):
        self._clear_rect()
        self._rect_id = self._canvas.create_rectangle(
            self._start_x, self._start_y, event.x, event.y,
            outline=self.OUTLINE_COLOR,
            width=2,
            fill=self.FILL_COLOR,
            stipple="gray25",
        )
        w = abs(event.x - self._start_x)
        h = abs(event.y - self._start_y)
        x1 = min(self._start_x, event.x)
        y1 = min(self._start_y, event.y)
        if self._coord_id:
            self._canvas.delete(self._coord_id)
        self._coord_id = self._canvas.create_text(
            x1 + 4, y1 + 4,
            text=f"{w} × {h}",
            fill=self.OUTLINE_COLOR,
            font=("Consolas", 11),
            anchor="nw",
        )

    def _on_release(self, event: tk.Event):
        x1 = min(self._start_x, event.x)
        y1 = min(self._start_y, event.y)
        x2 = max(self._start_x, event.x)
        y2 = max(self._start_y, event.y)
        w, h = x2 - x1, y2 - y1
        self.destroy()
        if w > 10 and h > 10:
            self._callback((x1, y1, w, h))
        else:
            self._callback(None)

    def _cancel(self):
        self.destroy()
        self._callback(None)

    def _clear_rect(self):
        if self._rect_id:
            self._canvas.delete(self._rect_id)
            self._rect_id = None
        if self._coord_id:
            self._canvas.delete(self._coord_id)
            self._coord_id = None
