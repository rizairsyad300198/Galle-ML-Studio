"""
ui/pages/start_page.py
Halaman awal: New Project & Open Project.
"""

import customtkinter as ctk
from ui.widgets import _CTkInputDialog


class StartScreen:
    def __init__(self, app):
        self.app = app

    def render(self):
        self.container = ctk.CTkFrame(self.app, fg_color="transparent")
        self.container.place(
            relx=0.5, rely=0.5, anchor="center", relwidth=0.5, relheight=0.6
        )

        ctk.CTkLabel(
            self.container,
            text="🚀 Galle ML Studio",
            font=ctk.CTkFont(family="Arial", size=48, weight="bold"),
        ).pack(pady=(80, 30))

        ctk.CTkLabel(
            self.container,
            text="Professional Machine Learning Workspace",
            font=ctk.CTkFont(family="Arial", size=20),
        ).pack(pady=(0, 60))

        btn_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        btn_frame.pack(pady=20, fill="x", padx=80)

        ctk.CTkButton(
            btn_frame,
            text="📁 New Project",
            width=250,
            height=55,
            font=ctk.CTkFont(family="Arial", size=18, weight="bold"),
            fg_color="#3498db",
            command=self.new_project,
        ).pack(pady=15)

        ctk.CTkButton(
            btn_frame,
            text="📂 Open Project",
            width=250,
            height=55,
            font=ctk.CTkFont(family="Arial", size=18),
            command=self.app.open_project,
        ).pack(pady=15)

    def new_project(self):
        dialog = _CTkInputDialog(
            self.app, title="New Project", prompt="Enter project name:"
        )
        name = dialog.get_input()
        if name and name.strip():
            self.app.create_project(name.strip())
