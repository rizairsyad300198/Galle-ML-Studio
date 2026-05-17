"""
ui/pages/upload_page.py
Step 0 — Upload Dataset & Dataset Preview.
"""

import customtkinter as ctk
from tkinter import filedialog, ttk
from ui.widgets import BasePage


class UploadPage(BasePage):

    # ── Step 0: Welcome / upload prompt ──────────────────────────────────────
    def show_welcome(self):
        self.clear_content()
        if hasattr(self.app, "df") and self.app.df is not None:
            self.show_dataset_screen()
            return

        for i in range(20):
            self.main_area.grid_rowconfigure(i, weight=0)
        self.main_area.grid_rowconfigure(0, weight=1)
        self.main_area.grid_rowconfigure(4, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.main_area,
            text="Welcome to Galle ML Studio!",
            font=ctk.CTkFont(family="Arial", size=40, weight="bold"),
        ).grid(row=1, column=0, pady=(0, 16))

        ctk.CTkLabel(
            self.main_area,
            text="Upload dataset CSV Anda untuk memulai",
            font=ctk.CTkFont(family="Arial", size=20),
            text_color="gray60",
        ).grid(row=2, column=0, pady=(0, 48))

        ctk.CTkButton(
            self.main_area,
            text="📁  Upload Dataset",
            width=360,
            height=70,
            font=ctk.CTkFont(family="Arial", size=20, weight="bold"),
            fg_color="#27ae60",
            hover_color="#219a52",
            command=self.show_upload_dialog,
        ).grid(row=3, column=0)

        self._nav_buttons(show_back=False, show_next=False)

    # ── Upload dialog ─────────────────────────────────────────────────────────
    def show_upload_dialog(self):
        filepath = filedialog.askopenfilename(
            title="Select CSV Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if filepath:
            self.ws.show_loading_overlay("Membaca dataset...")
            self.app.update()
            self.app.upload_dataset(filepath)
            self.ws.hide_loading_overlay()
            self._go_to_main_step(self.ws.MAIN_STEP_UPLOAD)

    # ── Step 0b: Dataset Preview ──────────────────────────────────────────────
    def show_dataset_screen(self):
        self.clear_content()

        for i in range(20):
            self.main_area.grid_rowconfigure(i, weight=0)
        self.main_area.grid_rowconfigure(0, weight=0)
        self.main_area.grid_rowconfigure(1, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        header_frame = ctk.CTkFrame(self.main_area, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(30, 10), padx=20)
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=0)

        ctk.CTkLabel(
            header_frame,
            text="✅  Dataset Loaded Successfully!",
            font=ctk.CTkFont(family="Arial", size=32, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header_frame,
            text=f"📊 {self.app.df.shape[0]:,} baris  ·  {self.app.df.shape[1]} kolom",
            font=ctk.CTkFont(family="Arial", size=16),
            text_color="gray60",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        ctk.CTkButton(
            header_frame,
            text="🔄  Ganti Dataset",
            width=160,
            height=38,
            fg_color="#3d4045",
            hover_color="#4a4d52",
            font=ctk.CTkFont(family="Arial", size=14),
            command=self.show_upload_dialog,
        ).grid(row=0, column=1, rowspan=2, sticky="e")

        table_container = ctk.CTkFrame(
            self.main_area, fg_color="#1e1e1e", corner_radius=10
        )
        table_container.grid(row=1, column=0, sticky="nsew", pady=(10, 0), padx=20)
        table_container.grid_rowconfigure(0, weight=0)
        table_container.grid_rowconfigure(1, weight=1)
        table_container.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            table_container,
            text="Data Preview (15 baris pertama)",
            font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
            text_color="gray60",
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 6))

        tree_frame = ctk.CTkFrame(table_container, fg_color="transparent")
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "DataTable.Treeview",
            background="#2b2d30",
            foreground="white",
            rowheight=28,
            fieldbackground="#2b2d30",
            bordercolor="#3d4045",
            borderwidth=0,
            font=("Consolas", 12),
        )
        style.configure(
            "DataTable.Treeview.Heading",
            background="#1a1c1e",
            foreground="#3498db",
            font=("Arial", 12, "bold"),
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "DataTable.Treeview",
            background=[("selected", "#3498db")],
            foreground=[("selected", "white")],
        )
        style.map(
            "DataTable.Treeview.Heading",
            background=[("active", "#252729")],
        )

        preview_df = self.app.df.head(15)
        columns = list(preview_df.columns)

        tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            style="DataTable.Treeview",
        )

        col_width = max(
            100, min(200, (self.app.winfo_width() - 320) // max(len(columns), 1))
        )
        for col in columns:
            col_str = str(col)
            display = col_str if len(col_str) <= 18 else col_str[:16] + "…"
            tree.heading(col, text=display, anchor="w")
            tree.column(col, width=col_width, minwidth=80, anchor="w")

        tree.tag_configure("odd", background="#2b2d30")
        tree.tag_configure("even", background="#252729")

        for i, (_, row) in enumerate(preview_df.iterrows()):
            tag = "odd" if i % 2 == 0 else "even"
            values = [s[:28] + "…" if len(s := str(v)) > 30 else s for v in row]
            tree.insert("", "end", values=values, tags=(tag,))

        v_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        h_scroll = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        self._nav_buttons(
            show_back=False,
            next_label="Next: Select Task →",
            next_cmd=lambda: self._go_to_main_step(self.ws.MAIN_STEP_TASK_SELECTION),
        )
