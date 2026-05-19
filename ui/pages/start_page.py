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
        import os
        from PIL import Image

        self.container = ctk.CTkFrame(self.app, fg_color="transparent")
        self.container.place(
            relx=0.5, rely=0.5, anchor="center", relwidth=0.5, relheight=0.6
        )
        logo_path = os.path.join("assets", "logo_with_text_dark_mode.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join("assets", "logo.png")

        if os.path.exists(logo_path):
            img = Image.open(logo_path)

            # ambil ukuran asli
            original_width, original_height = img.size

            # tentukan lebar target
            target_width = 220

            # hitung tinggi otomatis agar proporsional
            target_height = int((original_height / original_width) * target_width)

            logo_img = ctk.CTkImage(
                light_image=img,
                dark_image=img,
                size=(target_width, target_height),
            )

            ctk.CTkLabel(
                self.container,
                image=logo_img,
                text="",
            ).pack(pady=(60, 8))

        else:
            ctk.CTkLabel(
                self.container,
                text="🚀 Galle ML Studio",
                font=ctk.CTkFont(family="Arial", size=48, weight="bold"),
            ).pack(pady=(80, 30))

        ctk.CTkLabel(
            self.container,
            text="Professional Machine Learning Workspace",
            font=ctk.CTkFont(family="Arial", size=20),
            text_color="gray60",
        ).pack(pady=(0, 50))

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
