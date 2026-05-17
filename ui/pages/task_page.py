"""
ui/pages/task_page.py
Step 1 — Select Task Type (Classification / Regression / Anomaly).
"""

import customtkinter as ctk
from ui.widgets import BasePage


class TaskPage(BasePage):

    def show_task_selection_screen(self):
        self.clear_content()

        for i in range(20):
            self.main_area.grid_rowconfigure(i, weight=0)
        self.main_area.grid_rowconfigure(0, weight=0)
        self.main_area.grid_rowconfigure(1, weight=0)
        self.main_area.grid_rowconfigure(2, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.main_area,
            text="🧠  Select Task Type",
            font=ctk.CTkFont(family="Arial", size=36, weight="bold"),
        ).grid(row=0, column=0, pady=(50, 8))

        ctk.CTkLabel(
            self.main_area,
            text="Pilih jenis tugas machine learning yang ingin Anda jalankan.",
            font=ctk.CTkFont(family="Arial", size=17),
            text_color="gray60",
        ).grid(row=1, column=0, pady=(0, 30))

        cards_frame = ctk.CTkFrame(self.main_area, fg_color="transparent")
        cards_frame.grid(row=2, column=0, pady=10, padx=60, sticky="nsew")
        cards_frame.grid_columnconfigure((0, 1, 2), weight=1)
        cards_frame.grid_rowconfigure(0, weight=1)

        TASKS = [
            {
                "key": "classification",
                "icon": "🏷️",
                "title": "Klasifikasi",
                "desc": "Prediksi kategori / label.\nContoh: spam atau bukan spam,\njenis penyakit, sentimen teks.",
                "color": "#3498db",
                "hover": "#2980b9",
            },
            {
                "key": "regression",
                "icon": "📈",
                "title": "Regresi",
                "desc": "Prediksi nilai angka kontinu.\nContoh: harga rumah,\nsuhu, pendapatan.",
                "color": "#2ecc71",
                "hover": "#27ae60",
            },
            {
                "key": "anomaly",
                "icon": "🚨",
                "title": "Deteksi Anomali",
                "desc": "Temukan data yang 'tidak normal'.\nContoh: transaksi fraud,\nkerusakan mesin, intrusi jaringan.",
                "color": "#e74c3c",
                "hover": "#c0392b",
            },
        ]

        for i, task in enumerate(TASKS):
            card = ctk.CTkFrame(cards_frame, corner_radius=14, fg_color="#2B2D30")
            card.grid(row=0, column=i, padx=12, pady=10, ipadx=10, ipady=10, sticky="nsew")
            card.grid_rowconfigure((0, 1, 2, 3), weight=0)
            card.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(card, text=task["icon"], font=ctk.CTkFont(size=44)).grid(
                row=0, column=0, pady=(24, 6)
            )
            ctk.CTkLabel(
                card,
                text=task["title"],
                font=ctk.CTkFont(family="Arial", size=20, weight="bold"),
            ).grid(row=1, column=0, pady=(0, 8))
            ctk.CTkLabel(
                card,
                text=task["desc"],
                font=ctk.CTkFont(family="Arial", size=13),
                text_color="gray60",
                justify="center",
                wraplength=190,
            ).grid(row=2, column=0, pady=(0, 18), padx=12)
            ctk.CTkButton(
                card,
                text=f"Pilih {task['title']}",
                width=180,
                height=46,
                font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
                fg_color=task["color"],
                hover_color=task["hover"],
                command=lambda k=task["key"]: self._select_task_and_proceed(k),
            ).grid(row=3, column=0, pady=(0, 22))

        self._nav_buttons(
            show_back=True,
            back_label="← Back: Upload",
            back_cmd=lambda: self._go_to_main_step(self.ws.MAIN_STEP_UPLOAD),
            show_next=False,
        )

    def _select_task_and_proceed(self, task_key: str):
        self.app.inferred_task = task_key
        self._go_to_main_step(self.ws.MAIN_STEP_CONFIGURE)
