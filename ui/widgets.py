"""
ui/widgets.py
Shared widgets & base class dipakai oleh semua pages.
"""

import customtkinter as ctk


class _CTkInputDialog(ctk.CTkToplevel):
    """Styled, centered input dialog (pengganti tkinter.simpledialog)."""

    def __init__(self, parent, title="Input", prompt=""):
        super().__init__(parent)
        self._result = None

        self.title(title)
        self.resizable(False, False)
        self.grab_set()
        self.transient(parent)
        self.lift()

        w, h = 420, 200
        parent.update_idletasks()
        x = parent.winfo_rootx() + parent.winfo_width() // 2 - w // 2
        y = parent.winfo_rooty() + parent.winfo_height() // 2 - h // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        ctk.CTkLabel(self, text=prompt, font=ctk.CTkFont(family="Arial", size=15)).pack(
            pady=(30, 10)
        )
        self._entry = ctk.CTkEntry(
            self, width=300, height=40, font=ctk.CTkFont(family="Arial", size=14)
        )
        self._entry.pack(pady=(0, 20))
        self._entry.focus()
        self._entry.bind("<Return>", lambda e: self._ok())
        self._entry.bind("<Escape>", lambda e: self._cancel())

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack()
        ctk.CTkButton(
            btn_frame,
            text="OK",
            width=110,
            height=36,
            fg_color="#3498db",
            hover_color="#2980b9",
            font=ctk.CTkFont(weight="bold"),
            command=self._ok,
        ).pack(side="left", padx=8)
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            width=110,
            height=36,
            fg_color="#555",
            hover_color="#666",
            command=self._cancel,
        ).pack(side="left", padx=8)

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _ok(self):
        self._result = self._entry.get()
        self.destroy()

    def _cancel(self):
        self._result = None
        self.destroy()

    def get_input(self):
        return self._result


class BasePage:
    """
    Base class untuk semua page di WorkspaceScreen.
    Setiap page menerima referensi ke WorkspaceScreen (ws) dan app.
    """

    def __init__(self, ws):
        self.ws = ws          # WorkspaceScreen instance
        self.app = ws.app     # GalleMLStudio instance

    # ── Shortcut helpers ─────────────────────────────────────────────────────
    @property
    def main_area(self):
        return self.ws.main_area

    @property
    def nav_area(self):
        return self.ws.nav_area

    def clear_content(self):
        self.ws.clear_content()

    def _nav_buttons(self, **kw):
        self.ws._nav_buttons(**kw)

    def _go_to_main_step(self, step_idx):
        self.ws._go_to_main_step(step_idx)
