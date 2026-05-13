import customtkinter as ctk
from tkinter import filedialog, simpledialog, messagebox, ttk
import pandas as pd
import os
import numpy as np

from core.trainer import get_available_models, ALL_MODELS_CONFIG


class _CTkInputDialog(ctk.CTkToplevel):
    """A styled, centered input dialog to replace tkinter's simpledialog."""

    def __init__(self, parent, title="Input", prompt=""):
        super().__init__(parent)
        self._result = None

        self.title(title)
        self.resizable(False, False)
        self.grab_set()
        self.transient(parent)
        self.lift()
        self.geometry("420x200")

        window_width = 420
        window_height = 200

        parent.update_idletasks()
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()

        x = parent_x + (parent_width // 2) - (window_width // 2)
        y = parent_y + (parent_height // 2) - (window_height // 2)
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

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


class WorkspaceScreen:
    def __init__(self, app):
        self.app = app
        self.content_frame = None
        self.training_overlay = None
        self.training_animation_job = None
        self.training_start_time = None
        self.loading_overlay = None
        self.loading_message_label = None
        self.loading_spinner_label = None
        self.loading_animation_job = None
        self.sidebar_steps = {}
        self.target_var = None
        self.model_vars = {}
        self.feature_vars = {}
        self.split_var = None
        self.train_btn = None
        self.target_warning_label = None
        self._target_next_btn = None

        self._current_main_step = 0
        self.MAIN_STEP_UPLOAD = 0
        self.MAIN_STEP_TASK_SELECTION = 1
        self.MAIN_STEP_CONFIGURE = 2
        self.MAIN_STEP_TRAIN = 3
        self.MAIN_STEP_RESULTS = 4
        self.MAIN_STEP_TRY_MODEL = 5
        self.MAIN_STEP_EXPORT_MODEL = 6

        self._current_sub_step = 0
        self._sub_step_frames = []
        self._sub_step_nav_buttons = []
        self._step_nav_buttons = []

    # ─────────────────────────────────────────────────────────────────────────
    #  RENDER & SETUP
    # ─────────────────────────────────────────────────────────────────────────
    def render(self):
        self.app.grid_rowconfigure(0, weight=0)
        self.app.grid_rowconfigure(1, weight=1)
        self.app.grid_columnconfigure(0, weight=0)
        self.app.grid_columnconfigure(1, weight=1)

        sidebar = ctk.CTkFrame(self.app, width=280, corner_radius=0, fg_color="#202225")
        sidebar.grid(row=1, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        self.content_frame = ctk.CTkFrame(self.app)
        self.content_frame.grid(
            row=1, column=1, sticky="nsew", padx=(0, 15), pady=(0, 10)
        )
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=0)
        self.content_frame.grid_columnconfigure(0, weight=1)

        self.main_area = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.main_area.grid(row=0, column=0, sticky="nsew")
        self.main_area.grid_columnconfigure(0, weight=1)
        for i in range(20):
            self.main_area.grid_rowconfigure(i, weight=1)

        self.nav_area = ctk.CTkFrame(
            self.content_frame, fg_color="transparent", height=70
        )
        self.nav_area.grid(row=1, column=0, sticky="ew", padx=10, pady=(10, 15))
        self.nav_area.grid_propagate(False)

        self._setup_sidebar(sidebar)
        self._setup_training_overlay()
        self._setup_loading_overlay()

        if hasattr(self.app, "training_results") and self.app.training_results:
            self.sidebar_steps["results"].configure(state="normal")
            self.sidebar_steps["try_model"].configure(state="normal")
            self.sidebar_steps["export_model"].configure(state="normal")
            self._go_to_main_step(self.MAIN_STEP_RESULTS)
        else:
            self._go_to_main_step(self.MAIN_STEP_UPLOAD)

    def _setup_loading_overlay(self):
        self.loading_overlay = ctk.CTkFrame(self.app, corner_radius=0)
        self.loading_overlay.grid_rowconfigure(0, weight=1)
        self.loading_overlay.grid_columnconfigure(0, weight=1)

        container = ctk.CTkFrame(self.loading_overlay, fg_color="transparent")
        container.place(relx=0.5, rely=0.5, anchor="center")

        self.loading_spinner_label = ctk.CTkLabel(
            container,
            text="⏳",
            font=ctk.CTkFont(family="Arial", size=60, weight="bold"),
        )
        self.loading_spinner_label.pack(pady=20)

        self.loading_message_label = ctk.CTkLabel(
            container,
            text="Loading...",
            font=ctk.CTkFont(family="Arial", size=32, weight="bold"),
        )
        self.loading_message_label.pack(pady=(0, 10))

    def _setup_training_overlay(self):
        self.training_overlay = ctk.CTkFrame(self.app, corner_radius=0)
        self.training_overlay.grid_rowconfigure(0, weight=1)
        self.training_overlay.grid_columnconfigure(0, weight=1)

        container = ctk.CTkFrame(self.training_overlay, fg_color="transparent")
        container.place(relx=0.5, rely=0.5, anchor="center")

        self.training_spinner_label = ctk.CTkLabel(
            container,
            text="⚡",
            font=ctk.CTkFont(family="Arial", size=60, weight="bold"),
        )
        self.training_spinner_label.pack(pady=20)

        ctk.CTkLabel(
            container,
            text="Training Models...",
            font=ctk.CTkFont(family="Arial", size=32, weight="bold"),
        ).pack(pady=(0, 10))

        self.training_time_label = ctk.CTkLabel(
            container,
            text="Elapsed time: 0.0s",
            font=ctk.CTkFont(family="Arial", size=16),
            text_color="gray70",
        )
        self.training_time_label.pack(pady=(0, 20))

        ctk.CTkButton(
            container,
            text="✖ Cancel Training",
            width=200,
            height=44,
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            font=ctk.CTkFont(family="Arial", size=15, weight="bold"),
            command=self._confirm_cancel_training,
        ).pack(pady=(10, 0))

    def _confirm_cancel_training(self):
        popup = ctk.CTkToplevel(self.app)
        popup.title("Cancel Training")
        popup.geometry("400x200")
        popup.grab_set()
        popup.resizable(False, False)

        self.app.update_idletasks()
        x = self.app.winfo_rootx() + self.app.winfo_width() // 2 - 200
        y = self.app.winfo_rooty() + self.app.winfo_height() // 2 - 100
        popup.geometry(f"400x200+{x}+{y}")

        ctk.CTkLabel(
            popup,
            text="Batalkan Training?",
            font=ctk.CTkFont(family="Arial", size=20, weight="bold"),
        ).pack(pady=(30, 8))
        ctk.CTkLabel(
            popup,
            text="Proses training akan dihentikan.\nHasil yang belum selesai akan hilang.",
            font=ctk.CTkFont(family="Arial", size=14),
            text_color="gray60",
            justify="center",
        ).pack(pady=(0, 20))

        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack()

        ctk.CTkButton(
            btn_frame,
            text="Ya, Cancel",
            width=140,
            height=40,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            font=ctk.CTkFont(weight="bold"),
            command=lambda: self._do_cancel_training(popup),
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            btn_frame,
            text="Lanjutkan",
            width=140,
            height=40,
            fg_color="#555",
            hover_color="#666",
            command=popup.destroy,
        ).pack(side="left", padx=8)

    def _do_cancel_training(self, popup):
        popup.destroy()
        self.app.cancel_training = True
        self.set_training_state(False)
        # Kembali ke configure, UI akan restore dari app state (selected_models,
        # selected_features, target_column, split dsb sudah tersimpan sebelum training)
        self._go_to_main_step(self.MAIN_STEP_CONFIGURE)

    def _setup_sidebar(self, sidebar):
        sidebar.grid_columnconfigure(0, weight=1)
        for r in range(10):
            sidebar.grid_rowconfigure(r, weight=0)
        sidebar.grid_rowconfigure(9, weight=1)

        ctk.CTkLabel(
            sidebar,
            text="🎯 ML Workflow",
            font=ctk.CTkFont(family="Arial", size=28, weight="bold"),
        ).grid(row=0, column=0, pady=(30, 40))

        self.sidebar_steps = {}
        steps = [
            ("upload", "📁 1. Upload Dataset", None),
            ("task_selection", "🧠 2. Select Task", None),
            ("configure", "⚙️ 3. Configure Training", None),
            ("train", "⚡ 4. Train Models", "#e74c3c"),
            ("results", "📊 5. View Results", None),
            ("try_model", "🧪 6. Try Model", None),
            ("export_model", "📦 7. Export Model", None),
        ]
        for row_idx, (key, text, color) in enumerate(steps, start=1):
            kw = dict(
                text=text,
                width=240,
                height=50,
                state="disabled",
                font=ctk.CTkFont(family="Arial", size=16),
                fg_color="#2b2d30",
                text_color="gray50",
                command=None,
            )
            if color:
                kw["fg_color"] = color
            btn = ctk.CTkButton(sidebar, **kw)
            btn.grid(row=row_idx, column=0, pady=8, padx=20, sticky="ew")
            self.sidebar_steps[key] = btn

        info_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        info_frame.grid(row=8, column=0, pady=30, padx=20, sticky="ew")
        info_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            info_frame,
            text=f"Project: {self.app.current_project or 'None'}",
            font=ctk.CTkFont(family="Arial", size=14),
        ).grid(row=0, column=0)

    # ─────────────────────────────────────────────────────────────────────────
    #  NAVIGATION
    # ─────────────────────────────────────────────────────────────────────────
    def _go_to_main_step(self, step_idx):
        self._current_main_step = step_idx
        self.clear_content()
        self._update_sidebar_highlight()

        if step_idx == self.MAIN_STEP_UPLOAD:
            self.show_welcome()
            current_phase_str = "upload"
        elif step_idx == self.MAIN_STEP_TASK_SELECTION:
            if not hasattr(self.app, "df") or self.app.df is None:
                messagebox.showwarning("Warning", "Please upload a dataset first.")
                self._go_to_main_step(self.MAIN_STEP_UPLOAD)
                return
            self.show_task_selection_screen()
            current_phase_str = "task_selection"
        elif step_idx == self.MAIN_STEP_CONFIGURE:
            if not hasattr(self.app, "df") or self.app.df is None:
                messagebox.showwarning(
                    "Warning",
                    "Please upload a dataset first before configuring training.",
                )
                self._go_to_main_step(self.MAIN_STEP_UPLOAD)
                return
            if not hasattr(self.app, "inferred_task") or not self.app.inferred_task:
                messagebox.showwarning("Warning", "Please select a task type first.")
                self._go_to_main_step(self.MAIN_STEP_TASK_SELECTION)
                return
            self.show_target_screen()
            current_phase_str = "configure"
        elif step_idx == self.MAIN_STEP_TRAIN:
            current_phase_str = "train"
        elif step_idx == self.MAIN_STEP_RESULTS:
            self.show_results_screen()
            current_phase_str = "results"
        elif step_idx == self.MAIN_STEP_TRY_MODEL:
            self.show_try_model_screen()
            current_phase_str = "try_model"
        elif step_idx == self.MAIN_STEP_EXPORT_MODEL:
            self.show_export_model_screen()
            current_phase_str = "export_model"
        else:
            current_phase_str = "unknown"

        self.app._save_current_state(current_phase_str)

    def _update_sidebar_highlight(self):
        step_names = [
            "upload",
            "task_selection",
            "configure",
            "train",
            "results",
            "try_model",
            "export_model",
        ]
        for i, name in enumerate(step_names):
            btn = self.sidebar_steps.get(name)
            if btn:
                if i == self._current_main_step:
                    btn.configure(fg_color="#3498db", text_color="white")
                else:
                    btn.configure(fg_color="#2b2d30", text_color="gray50")

    def set_training_state(self, is_training):
        if is_training:
            self.training_overlay.grid(row=1, column=1, sticky="nsew")
            self.training_overlay.tkraise()
            self.training_start_time = pd.Timestamp.now()
            self._update_training_animation()
            if self.train_btn and self.train_btn.winfo_exists():
                self.train_btn.configure(state="disabled")
        else:
            if self.training_animation_job:
                self.training_overlay.after_cancel(self.training_animation_job)
                self.training_animation_job = None
            self.training_overlay.grid_forget()
            if self.train_btn and self.train_btn.winfo_exists():
                self.train_btn.configure(state="normal")

    def _update_training_animation(self):
        if not self.training_start_time:
            return
        elapsed = pd.Timestamp.now() - self.training_start_time
        self.training_time_label.configure(
            text=f"Elapsed time: {elapsed.total_seconds():.1f}s"
        )
        spinner_chars = ["⚡", "🔥", "🤖", "🧠", "⚙️"]
        idx = int(elapsed.total_seconds() * 2) % len(spinner_chars)
        self.training_spinner_label.configure(text=spinner_chars[idx])
        self.training_animation_job = self.training_overlay.after(
            150, self._update_training_animation
        )

    def show_loading_overlay(self, message="Loading..."):
        self.loading_overlay.grid(row=1, column=1, sticky="nsew")
        self.loading_overlay.tkraise()
        self.loading_message_label.configure(text=message)
        self._update_loading_animation_spinner()

    def hide_loading_overlay(self):
        if self.loading_animation_job:
            self.loading_overlay.after_cancel(self.loading_animation_job)
            self.loading_animation_job = None
        self.loading_overlay.grid_forget()

    def _update_loading_animation_spinner(self):
        spinner_chars = ["⏳", "⌛"]
        idx = int(pd.Timestamp.now().timestamp() * 5) % len(spinner_chars)
        self.loading_spinner_label.configure(text=spinner_chars[idx])
        self.loading_animation_job = self.loading_overlay.after(
            200, self._update_loading_animation_spinner
        )

    def clear_content(self):
        for widget in self.main_area.winfo_children():
            widget.destroy()
        for widget in self.nav_area.winfo_children():
            widget.destroy()

    # ─────────────────────────────────────────────────────────────────────────
    #  UNIFIED NAV HELPER
    # ─────────────────────────────────────────────────────────────────────────
    def _nav_buttons(
        self,
        *,
        show_back=True,
        show_next=True,
        next_label="Next →",
        next_cmd=None,
        back_cmd=None,
        back_label="← Back",
    ):
        bar = self.nav_area
        for widget in bar.winfo_children():
            widget.destroy()

        bar.grid_columnconfigure(0, weight=1)
        bar.grid_columnconfigure(1, weight=1)

        if show_back:
            ctk.CTkButton(
                bar,
                text=back_label,
                width=180,
                height=44,
                fg_color="#3d4045",
                hover_color="#4a4d52",
                font=ctk.CTkFont(family="Arial", size=15),
                command=back_cmd,
            ).grid(row=0, column=0, sticky="w", padx=(0, 4))

        if show_next:
            ctk.CTkButton(
                bar,
                text=next_label,
                width=200,
                height=44,
                fg_color="#3498db",
                hover_color="#2980b9",
                font=ctk.CTkFont(family="Arial", size=15, weight="bold"),
                command=next_cmd,
            ).grid(row=0, column=1, sticky="e", padx=(4, 0))

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 0 — UPLOAD
    # ─────────────────────────────────────────────────────────────────────────
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

    def show_upload_dialog(self):
        filepath = filedialog.askopenfilename(
            title="Select CSV Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if filepath:
            self.show_loading_overlay("Membaca dataset...")
            self.app.update()
            self.app.upload_dataset(filepath)
            self.hide_loading_overlay()
            self._go_to_main_step(self.MAIN_STEP_UPLOAD)

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 0b — DATASET PREVIEW
    # ─────────────────────────────────────────────────────────────────────────
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
            values = []
            for v in row:
                s = str(v)
                values.append(s if len(s) <= 30 else s[:28] + "…")
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
            next_cmd=lambda: self._go_to_main_step(self.MAIN_STEP_TASK_SELECTION),
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 1 — TASK SELECTION
    # ─────────────────────────────────────────────────────────────────────────
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
            card.grid(
                row=0, column=i, padx=12, pady=10, ipadx=10, ipady=10, sticky="nsew"
            )
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
            back_cmd=lambda: self._go_to_main_step(self.MAIN_STEP_UPLOAD),
            show_next=False,
        )

    def _select_task_and_proceed(self, task_key: str):
        self.app.inferred_task = task_key
        self._go_to_main_step(self.MAIN_STEP_CONFIGURE)

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 2 — CONFIGURE TRAINING (sub-wizard)
    # ─────────────────────────────────────────────────────────────────────────
    def show_target_screen(self):
        self.clear_content()
        self._current_sub_step = 0

        is_anomaly = getattr(self.app, "inferred_task", "") == "anomaly"

        for i in range(20):
            self.main_area.grid_rowconfigure(i, weight=0)
        self.main_area.grid_rowconfigure(0, weight=0)
        self.main_area.grid_rowconfigure(1, weight=0)
        self.main_area.grid_rowconfigure(2, weight=0)
        self.main_area.grid_rowconfigure(3, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        title_bar = ctk.CTkFrame(self.main_area, fg_color="transparent")
        title_bar.grid(row=0, column=0, sticky="ew", pady=(20, 0), padx=30)
        title_bar.grid_columnconfigure(0, weight=1)
        task_label = getattr(self.app, "inferred_task", "").capitalize()
        ctk.CTkLabel(
            title_bar,
            text=f"Configure Training  —  {task_label}",
            font=ctk.CTkFont(family="Arial", size=32, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        if is_anomaly:
            step_labels = [("1", "Features"), ("2", "Models"), ("3", "Split & Run")]
        else:
            step_labels = [
                ("1", "Target"),
                ("2", "Features"),
                ("3", "Models"),
                ("4", "Split & Run"),
            ]

        tab_bar = ctk.CTkFrame(self.main_area, fg_color="transparent")
        tab_bar.grid(row=1, column=0, sticky="ew", padx=30, pady=(10, 0))
        for i in range(len(step_labels)):
            tab_bar.grid_columnconfigure(i, weight=1)

        self._step_nav_buttons = []
        for i, (num, label) in enumerate(step_labels):
            btn = ctk.CTkButton(
                tab_bar,
                text=f"  {num}. {label}  ",
                width=140,
                height=38,
                corner_radius=8,
                font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
                command=lambda idx=i: self._try_go_to_step(idx),
            )
            btn.grid(row=0, column=i, padx=(0, 6), sticky="ew")
            self._step_nav_buttons.append(btn)

        sep = ctk.CTkFrame(self.main_area, height=2, fg_color="#3d4045")
        sep.grid(row=2, column=0, sticky="ew", padx=30, pady=(8, 0))

        self._wizard_area = ctk.CTkFrame(self.main_area, fg_color="transparent")
        self._wizard_area.grid(row=3, column=0, sticky="nsew", padx=30, pady=(10, 10))
        self._wizard_area.grid_columnconfigure(0, weight=1)
        self._wizard_area.grid_rowconfigure(0, weight=1)

        self._placeholder = "-- Select Target --"

        # ── RESTORE target_var dari app state ─────────────────────────────────
        saved_target = getattr(self.app, "target_column", None) or self._placeholder
        self.target_var = ctk.StringVar(
            value=saved_target if not is_anomaly else self._placeholder
        )

        # ── RESTORE split_var dari app state ──────────────────────────────────
        saved_split = getattr(self.app, "train_test_split", "80/20 (Recommended)")
        self.split_var = ctk.StringVar(value=saved_split)

        # target dianggap sudah dipilih jika ada saved value
        self._target_selected = is_anomaly or (
            saved_target and saved_target != self._placeholder
        )

        # ── Jika anomaly, langsung ke fitur; jika tidak, mulai dari target ────
        if is_anomaly:
            self._go_to_step(0)
        else:
            self._go_to_step(0)

    def _try_go_to_step(self, idx: int):
        is_anomaly = getattr(self.app, "inferred_task", "") == "anomaly"
        if not is_anomaly and idx > 0 and not getattr(self, "_target_selected", False):
            messagebox.showwarning(
                "Pilih Target Dulu",
                "Silakan pilih kolom target sebelum melanjutkan ke langkah berikutnya.",
            )
            return
        self._go_to_step(idx)

    def _go_to_step(self, idx: int):
        is_anomaly = getattr(self.app, "inferred_task", "") == "anomaly"
        features_step_idx = 0 if is_anomaly else 1
        models_step_idx = 1 if is_anomaly else 2

        if self._current_sub_step == features_step_idx:
            self.app.selected_features = [
                col for col, var in self.feature_vars.items() if var.get()
            ]

        if self._current_sub_step == models_step_idx:
            user_params = {}
            for model_name, is_selected_var in self.model_vars.items():
                if is_selected_var.get():
                    model_config = ALL_MODELS_CONFIG.get(model_name)
                    if not model_config:
                        continue

                    model_specific_params = {}
                    for param_name, param_widget_var in self.model_param_widgets.get(
                        model_name, {}
                    ).items():
                        param_config = model_config["params"].get(param_name)
                        if not param_config:
                            continue

                        value = param_widget_var.get()

                        try:
                            if param_config["type"] == "bool":
                                model_specific_params[param_name] = bool(value)
                            elif param_config["type"] == "int":
                                if param_config.get("none_option") and value == "None":
                                    model_specific_params[param_name] = None
                                else:
                                    val = int(value)
                                    if (
                                        "min" in param_config
                                        and val < param_config["min"]
                                    ):
                                        raise ValueError(
                                            f"Min value is {param_config['min']}"
                                        )
                                    if (
                                        "max" in param_config
                                        and val > param_config["max"]
                                    ):
                                        raise ValueError(
                                            f"Max value is {param_config['max']}"
                                        )
                                    model_specific_params[param_name] = val
                            elif param_config["type"] == "float":
                                if param_config.get("none_option") and value == "None":
                                    model_specific_params[param_name] = None
                                else:
                                    val = float(value)
                                    if (
                                        "min" in param_config
                                        and val < param_config["min"]
                                    ):
                                        raise ValueError(
                                            f"Min value is {param_config['min']}"
                                        )
                                    if (
                                        "max" in param_config
                                        and val > param_config["max"]
                                    ):
                                        raise ValueError(
                                            f"Max value is {param_config['max']}"
                                        )
                                    model_specific_params[param_name] = val
                            elif param_config["type"] == "str":
                                if param_config.get("none_option") and value == "None":
                                    model_specific_params[param_name] = None
                                else:
                                    model_specific_params[param_name] = value
                        except ValueError as e:
                            messagebox.showerror(
                                "Input Error",
                                f"Invalid value for {model_name} - {param_name}: {value}. {e}",
                            )
                            return
                        except Exception as e:
                            messagebox.showerror(
                                "Input Error",
                                f"Error processing {model_name} - {param_name}: {e}",
                            )
                            return
                    user_params[model_name] = model_specific_params
            self.app.user_model_params = user_params

        self._current_sub_step = idx
        self._update_tab_highlight()

        for w in self._wizard_area.winfo_children():
            w.destroy()
        self._step_frames = []

        if is_anomaly:
            builders = [
                self._build_step_features,
                self._build_step_models,
                self._build_step_split,
            ]
        else:
            builders = [
                self._build_step_target,
                self._build_step_features,
                self._build_step_models,
                self._build_step_split,
            ]

        builders[idx](self._wizard_area)
        self.app._save_current_state("configure")

    def _update_tab_highlight(self):
        for i, btn in enumerate(self._step_nav_buttons):
            if i == self._current_sub_step:
                btn.configure(fg_color="#3498db", text_color="white")
            else:
                btn.configure(fg_color="#2b2d30", text_color="gray70")

    # ── Sub-step 0: TARGET ────────────────────────────────────────────────────
    def _build_step_target(self, parent):
        for i in range(10):
            parent.grid_rowconfigure(i, weight=0)
        parent.grid_rowconfigure(5, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        columns = list(self.app.df.columns)

        ctk.CTkLabel(
            parent,
            text="Step 1  —  Select Target Column",
            font=ctk.CTkFont(family="Arial", size=22, weight="bold"),
        ).grid(row=0, column=0, sticky="w", pady=(10, 4))
        ctk.CTkLabel(
            parent,
            text="Kolom 'jawaban' yang akan coba diprediksi oleh model.",
            font=ctk.CTkFont(family="Arial", size=13),
            text_color="gray60",
        ).grid(row=1, column=0, sticky="w", pady=(0, 12))

        target_menu = ctk.CTkOptionMenu(
            parent,
            variable=self.target_var,
            values=[self._placeholder] + columns,
            width=400,
            font=ctk.CTkFont(family="Arial", size=15),
            command=self._on_target_changed,
        )
        target_menu.grid(row=2, column=0, sticky="w")

        self.target_warning_label = ctk.CTkLabel(
            parent,
            text="",
            text_color="#E74C3C",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.target_warning_label.grid(row=3, column=0, sticky="w", pady=(6, 0))

        ctk.CTkLabel(
            parent,
            text="Distribution preview:",
            font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
        ).grid(row=4, column=0, sticky="w", pady=(18, 4))

        self.target_preview = ctk.CTkScrollableFrame(parent)
        self.target_preview.grid(row=5, column=0, sticky="nsew")
        self.target_preview.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.target_preview,
            text="Pilih kolom target terlebih dahulu...",
            text_color="gray60",
            font=ctk.CTkFont(family="Arial", size=13),
        ).grid(row=0, column=0, pady=20)

        self._nav_buttons(
            show_back=True,
            back_label="← Back: Select Task",
            back_cmd=lambda: self._go_to_main_step(self.MAIN_STEP_TASK_SELECTION),
            show_next=True,
            next_label="Next: Features →",
            next_cmd=lambda: self._try_go_to_step(1),
        )
        self._target_next_btn = None
        for w in self.nav_area.winfo_children():
            try:
                if "Features" in str(w.cget("text")):
                    w.configure(state="disabled")
                    self._target_next_btn = w
                    break
            except Exception:
                pass

        # ── Jika target sudah dipilih sebelumnya (restore state), load preview ──
        current_val = self.target_var.get()
        if (
            current_val
            and current_val != self._placeholder
            and current_val in self.app.df.columns
        ):
            self._on_target_changed(current_val)

    def _on_target_changed(self, value=None, *_):
        selected = self.target_var.get()
        if not selected or selected == self._placeholder:
            self._target_selected = False
            if self._target_next_btn and self._target_next_btn.winfo_exists():
                self._target_next_btn.configure(state="disabled")
            if hasattr(self, "target_preview") and self.target_preview.winfo_exists():
                for w in self.target_preview.winfo_children():
                    w.destroy()
                ctk.CTkLabel(
                    self.target_preview,
                    text="Pilih kolom target terlebih dahulu...",
                    text_color="gray60",
                    font=ctk.CTkFont(family="Arial", size=13),
                ).pack(pady=20)
            return

        self._target_selected = True
        self._update_target_preview_and_task()
        self._validate_target_column()

    def _update_target_preview_and_task(self):
        if not self.target_var:
            return
        col = self.target_var.get()
        if not col or col == self._placeholder or col not in self.app.df.columns:
            return

        value_counts = self.app.df[col].value_counts().head(10)
        total = value_counts.sum()

        if hasattr(self, "target_preview") and self.target_preview.winfo_exists():
            for w in self.target_preview.winfo_children():
                w.destroy()

            header = ctk.CTkFrame(self.target_preview, fg_color="transparent")
            header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
            header.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                header,
                text=f"Target: {col}  |  {len(self.app.df[col].unique())} unique values  |  {len(self.app.df)} rows",
                font=ctk.CTkFont(family="Arial", size=12),
                text_color="gray60",
            ).grid(row=0, column=0, sticky="w")

            sep = ctk.CTkFrame(self.target_preview, height=1, fg_color="#3d4045")
            sep.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

            bar_colors = [
                "#3498db",
                "#2ecc71",
                "#e74c3c",
                "#f39c12",
                "#9b59b6",
                "#1abc9c",
                "#e67e22",
                "#34495e",
                "#e91e63",
                "#00bcd4",
            ]

            for i, (label, count) in enumerate(value_counts.items()):
                pct = count / total * 100
                color = bar_colors[i % len(bar_colors)]
                row_idx = i + 2
                self.target_preview.grid_rowconfigure(row_idx, weight=0)

                row = ctk.CTkFrame(self.target_preview, fg_color="transparent")
                row.grid(row=row_idx, column=0, sticky="ew", padx=10, pady=3)
                row.grid_columnconfigure(0, weight=0)
                row.grid_columnconfigure(1, weight=1)
                row.grid_columnconfigure(2, weight=0)

                label_str = str(label)
                display = label_str if len(label_str) <= 35 else label_str[:32] + "..."
                ctk.CTkLabel(
                    row,
                    text=display,
                    width=240,
                    anchor="w",
                    font=ctk.CTkFont(family="Arial", size=12),
                ).grid(row=0, column=0, sticky="w")

                bar_track = ctk.CTkFrame(
                    row, height=18, fg_color="#2b2d30", corner_radius=4
                )
                bar_track.grid(row=0, column=1, sticky="ew", padx=(8, 8))
                bar_track.grid_propagate(False)

                bar_fill = ctk.CTkFrame(
                    bar_track,
                    height=18,
                    fg_color=color,
                    corner_radius=4,
                    width=max(4, int(pct / 100 * 280)),
                )
                bar_fill.grid(row=0, column=0, sticky="w")

                ctk.CTkLabel(
                    row,
                    text=f"{count:,}  ({pct:.1f}%)",
                    width=100,
                    anchor="e",
                    font=ctk.CTkFont(family="Arial", size=12),
                    text_color="gray70",
                ).grid(row=0, column=2, sticky="e")

        if getattr(self.app, "inferred_task", "") not in (
            "classification",
            "regression",
        ):
            target_data = self.app.df[col]
            if (
                pd.api.types.is_numeric_dtype(target_data)
                and target_data.nunique() > 20
            ):
                self.app.inferred_task = "regression"
            else:
                self.app.inferred_task = "classification"

    def _validate_target_column(self):
        if not self.target_var:
            return
        target_col = self.target_var.get()

        if (
            not target_col
            or target_col == self._placeholder
            or target_col not in self.app.df.columns
        ):
            if self._target_next_btn and self._target_next_btn.winfo_exists():
                self._target_next_btn.configure(state="disabled")
            return

        target_data = self.app.df[target_col]
        is_suitable = True
        warning_text = ""

        if not pd.api.types.is_numeric_dtype(target_data):
            if target_data.nunique() / len(target_data) > 0.5:
                is_suitable = False
                warning_text = "Peringatan: Terlalu banyak nilai unik — kolom ini mungkin tidak cocok sebagai target."

        if "date" in target_col.lower() or "tgl" in target_col.lower():
            try:
                pd.to_datetime(target_data, errors="raise")
                is_suitable = False
                warning_text = "Peringatan: Kolom tanggal tidak cocok sebagai target."
            except (ValueError, TypeError):
                pass

        if (
            hasattr(self, "target_warning_label")
            and self.target_warning_label.winfo_exists()
        ):
            self.target_warning_label.configure(
                text=warning_text if not is_suitable else ""
            )

        if self._target_next_btn and self._target_next_btn.winfo_exists():
            self._target_next_btn.configure(
                state="normal" if is_suitable else "disabled"
            )

    # ── Sub-step 1: FEATURES ──────────────────────────────────────────────────
    def _build_step_features(self, parent):
        for i in range(10):
            parent.grid_rowconfigure(i, weight=0)
        parent.grid_rowconfigure(3, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        is_anomaly = getattr(self.app, "inferred_task", "") == "anomaly"
        step_num = "1" if is_anomaly else "2"

        ctk.CTkLabel(
            parent,
            text=f"Step {step_num}  —  Select Features",
            font=ctk.CTkFont(family="Arial", size=22, weight="bold"),
        ).grid(row=0, column=0, sticky="w", pady=(10, 4))
        ctk.CTkLabel(
            parent,
            text="Pilih kolom-kolom yang akan digunakan model untuk memprediksi target.",
            font=ctk.CTkFont(family="Arial", size=13),
            text_color="gray60",
        ).grid(row=1, column=0, sticky="w", pady=(0, 10))

        qa = ctk.CTkFrame(parent, fg_color="transparent")
        qa.grid(row=2, column=0, sticky="w", pady=(0, 8))
        ctk.CTkButton(
            qa,
            text="✅ Pilih Semua",
            width=130,
            height=32,
            command=lambda: self._toggle_all_features(True),
        ).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkButton(
            qa,
            text="☐ Batalkan Semua",
            width=140,
            height=32,
            fg_color="#555",
            hover_color="#666",
            command=lambda: self._toggle_all_features(False),
        ).grid(row=0, column=1)

        scroll = ctk.CTkScrollableFrame(parent)
        scroll.grid(row=3, column=0, sticky="nsew")
        scroll.grid_columnconfigure((0, 1), weight=1)

        target_col = self.target_var.get() if not is_anomaly else ""
        self.feature_vars = {}
        non_target_cols = [c for c in self.app.df.columns if c != target_col]

        # ── RESTORE: gunakan selected_features dari app state jika ada ────────
        saved_features = set(getattr(self.app, "selected_features", []) or [])
        # Jika belum pernah dipilih, default semua kolom dipilih
        initial_selection = saved_features if saved_features else set(non_target_cols)

        for i, col in enumerate(non_target_cols):
            var = ctk.BooleanVar(value=(col in initial_selection))
            ctk.CTkCheckBox(
                scroll,
                text=col,
                variable=var,
                font=ctk.CTkFont(family="Arial", size=14),
                checkbox_width=20,
                checkbox_height=20,
            ).grid(row=i // 2, column=i % 2, sticky="w", pady=4, padx=10)
            self.feature_vars[col] = var

        back_label = "← Back: Select Task" if is_anomaly else "← Back: Target"
        back_cmd = lambda: (
            self._go_to_main_step(self.MAIN_STEP_TASK_SELECTION)
            if is_anomaly
            else self._go_to_step(0)
        )
        self._nav_buttons(
            show_back=True,
            back_label=back_label,
            back_cmd=back_cmd,
            next_label="Next: Models →",
            next_cmd=lambda: self._go_to_step(1 if is_anomaly else 2),
        )

    def _toggle_all_features(self, select_all):
        for var in self.feature_vars.values():
            var.set(select_all)

    # ── Sub-step 2: MODELS ────────────────────────────────────────────────────
    def _build_step_models(self, parent):
        is_anomaly = getattr(self.app, "inferred_task", "") == "anomaly"
        step_num = "2" if is_anomaly else "3"

        ctk.CTkLabel(
            parent,
            text=f"Step {step_num}  —  Select Models",
            font=ctk.CTkFont(family="Arial", size=22, weight="bold"),
        ).pack(anchor="w", pady=(10, 4))
        ctk.CTkLabel(
            parent,
            text="Pilih model yang akan dilatih dan atur parameternya.",
            font=ctk.CTkFont(family="Arial", size=13),
            text_color="gray60",
        ).pack(anchor="w", pady=(0, 16))

        trainable_classification = get_available_models("classification")
        trainable_regression = get_available_models("regression")
        trainable_anomaly = get_available_models("anomaly")

        self.model_vars = {}
        self.model_param_widgets = {}
        self.model_param_frames = {}

        # ── RESTORE selected_models dari app state ────────────────────────────
        saved_selected_models = set(getattr(self.app, "selected_models", []) or [])
        saved_user_params = getattr(self.app, "user_model_params", {}) or {}

        scroll_frame = ctk.CTkScrollableFrame(parent)
        scroll_frame.pack(fill="both", expand=True)

        models_by_task_type = {
            "classification": [],
            "regression": [],
            "anomaly": [],
            "other": [],
        }
        for model_name, config in ALL_MODELS_CONFIG.items():
            if model_name in trainable_classification:
                models_by_task_type["classification"].append((model_name, config))
            elif model_name in trainable_regression:
                models_by_task_type["regression"].append((model_name, config))
            elif model_name in trainable_anomaly:
                models_by_task_type["anomaly"].append((model_name, config))
            else:
                models_by_task_type["other"].append((model_name, config))

        display_categories = []
        if is_anomaly:
            display_categories.append(
                ("Deteksi Anomali", models_by_task_type["anomaly"])
            )
        else:
            display_categories.append(
                ("Klasifikasi", models_by_task_type["classification"])
            )
            display_categories.append(("Regresi", models_by_task_type["regression"]))

        if models_by_task_type["other"]:
            display_categories.append(
                ("Lain-lain (Belum Didukung)", models_by_task_type["other"])
            )

        for category_title, models_in_category in display_categories:
            if not models_in_category:
                continue

            ctk.CTkLabel(
                scroll_frame,
                text=category_title,
                font=ctk.CTkFont(
                    family="Arial", size=18, weight="bold", underline=True
                ),
            ).pack(anchor="w", padx=10, pady=(20, 10))

            for model_name, config in models_in_category:
                is_trainable_for_task = (
                    is_anomaly and model_name in trainable_anomaly
                ) or (
                    not is_anomaly
                    and (
                        model_name in trainable_classification
                        or model_name in trainable_regression
                    )
                )

                # ── RESTORE: centang model yang sebelumnya dipilih ─────────────
                was_selected = (
                    model_name in saved_selected_models
                    if saved_selected_models
                    else False
                )
                var = ctk.BooleanVar(value=(was_selected and is_trainable_for_task))
                self.model_vars[model_name] = var

                model_card = ctk.CTkFrame(
                    scroll_frame, corner_radius=8, fg_color="#2B2D30"
                )
                model_card.pack(fill="x", padx=10, pady=5)
                model_card.grid_columnconfigure(0, weight=1)

                checkbox = ctk.CTkCheckBox(
                    model_card,
                    text=model_name,
                    variable=var,
                    font=ctk.CTkFont(family="Arial", size=15, weight="bold"),
                    checkbox_width=24,
                    checkbox_height=24,
                    command=lambda m=model_name: self._toggle_model_params_visibility(
                        m
                    ),
                )
                checkbox.grid(row=0, column=0, sticky="w", padx=20, pady=(15, 0))
                if not is_trainable_for_task:
                    checkbox.configure(state="disabled")
                    var.set(False)

                param_frame = ctk.CTkFrame(model_card, fg_color="transparent")
                param_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 15))
                param_frame.grid_columnconfigure(0, weight=1)
                self.model_param_frames[model_name] = param_frame

                if not var.get() or not config["params"]:
                    param_frame.grid_remove()

                self.model_param_widgets[model_name] = {}

                row_idx = 0
                for param_name, param_config in config["params"].items():
                    if param_config.get("hidden", False):
                        continue

                    # ── RESTORE param values dari app state ───────────────────
                    current_value = saved_user_params.get(model_name, {}).get(
                        param_name,
                        self.app.user_model_params.get(model_name, {}).get(
                            param_name, param_config["default"]
                        ),
                    )

                    param_label = ctk.CTkLabel(
                        param_frame,
                        text=f"{param_name.replace('_', ' ').title()}:",
                        font=ctk.CTkFont(size=13, weight="bold"),
                        anchor="w",
                    )
                    param_label.grid(row=row_idx, column=0, sticky="w", pady=(5, 2))

                    if param_config["type"] == "bool":
                        param_var = ctk.BooleanVar(value=current_value)
                        widget = ctk.CTkCheckBox(
                            param_frame, text="", variable=param_var
                        )
                        widget.grid(row=row_idx, column=1, sticky="w", pady=(5, 2))
                        self.model_param_widgets[model_name][param_name] = param_var
                    elif param_config["type"] in ("int", "float"):
                        param_var = ctk.StringVar(value=str(current_value))
                        widget = ctk.CTkEntry(
                            param_frame, textvariable=param_var, width=150
                        )
                        widget.grid(row=row_idx, column=1, sticky="ew", pady=(5, 2))
                        self.model_param_widgets[model_name][param_name] = param_var
                    elif param_config["type"] == "str":
                        if "options" in param_config:
                            options = [str(opt) for opt in param_config["options"]]
                            if param_config.get("none_option"):
                                options = ["None"] + options
                            param_var = ctk.StringVar(value=str(current_value))
                            widget = ctk.CTkOptionMenu(
                                param_frame, values=options, variable=param_var
                            )
                            widget.grid(row=row_idx, column=1, sticky="ew", pady=(5, 2))
                            self.model_param_widgets[model_name][param_name] = param_var
                        else:
                            param_var = ctk.StringVar(value=str(current_value))
                            widget = ctk.CTkEntry(
                                param_frame, textvariable=param_var, width=150
                            )
                            widget.grid(row=row_idx, column=1, sticky="ew", pady=(5, 2))
                            self.model_param_widgets[model_name][param_name] = param_var
                    row_idx += 1

                ctk.CTkLabel(
                    model_card,
                    text="Atur parameter model di atas.",
                    font=ctk.CTkFont(family="Arial", size=12),
                    text_color="gray70",
                    wraplength=700,
                    justify="left",
                ).grid(row=2, column=0, sticky="w", padx=20, pady=(0, 15))

        features_step_idx = 0 if is_anomaly else 1
        split_step_idx = 2 if is_anomaly else 3

        self._nav_buttons(
            show_back=True,
            back_label="← Back: Features",
            back_cmd=lambda: self._go_to_step(features_step_idx),
            next_label="Next: Split & Run →",
            next_cmd=lambda: self._go_to_step(split_step_idx),
        )

    def _toggle_model_params_visibility(self, model_name):
        if self.model_vars[model_name].get():
            if self.model_param_frames[model_name]:
                self.model_param_frames[model_name].grid()
        else:
            if self.model_param_frames[model_name]:
                self.model_param_frames[model_name].grid_remove()

    # ── Sub-step 3: SPLIT + LAUNCH ────────────────────────────────────────────
    def _build_step_split(self, parent):
        for i in range(10):
            parent.grid_rowconfigure(i, weight=0)
        parent.grid_rowconfigure(5, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        is_anomaly = getattr(self.app, "inferred_task", "") == "anomaly"
        step_num = "3" if is_anomaly else "4"

        ctk.CTkLabel(
            parent,
            text=f"Step {step_num}  —  Train / Test Split & Launch",
            font=ctk.CTkFont(family="Arial", size=22, weight="bold"),
        ).grid(row=0, column=0, sticky="w", pady=(10, 4))
        ctk.CTkLabel(
            parent,
            text="Membagi data: sebagian untuk 'belajar' (Training), sebagian untuk 'ujian' (Testing).",
            font=ctk.CTkFont(family="Arial", size=13),
            text_color="gray60",
        ).grid(row=1, column=0, sticky="w", pady=(0, 20))

        ctk.CTkOptionMenu(
            parent,
            variable=self.split_var,
            values=["70/30", "80/20 (Recommended)", "90/10"],
            width=300,
            font=ctk.CTkFont(family="Arial", size=15),
        ).grid(row=2, column=0, sticky="w")

        sep = ctk.CTkFrame(parent, height=2, fg_color="#3d4045")
        sep.grid(row=3, column=0, sticky="ew", pady=24)

        ctk.CTkLabel(
            parent,
            text=">> Training Summary",
            font=ctk.CTkFont(family="Arial", size=17, weight="bold"),
        ).grid(row=4, column=0, sticky="w", pady=(0, 8))

        self._summary_box = ctk.CTkTextbox(
            parent, height=130, font=("Consolas", 12), border_spacing=12
        )
        self._summary_box.grid(row=5, column=0, sticky="ew")
        self._refresh_summary()

        self.train_btn = ctk.CTkButton(
            parent,
            text="🚀  START TRAINING",
            width=360,
            height=64,
            font=ctk.CTkFont(family="Arial", size=20, weight="bold"),
            fg_color="#e74c3c",
            hover_color="#c0392b",
            command=self.confirm_and_start_training,
        )
        self.train_btn.grid(row=6, column=0, pady=(20, 10))

        models_step_idx = 1 if is_anomaly else 2
        self._nav_buttons(
            show_back=True,
            back_label="← Back: Models",
            back_cmd=lambda: self._go_to_step(models_step_idx),
            show_next=False,
        )

    def _refresh_summary(self):
        if not hasattr(self, "_summary_box") or not self._summary_box.winfo_exists():
            return

        task = getattr(self.app, "inferred_task", "—")
        target = self.target_var.get() if self.target_var else "—"
        is_anomaly = task == "anomaly"
        n_feat = sum(1 for v in self.feature_vars.values() if v.get())
        selected_models = [k for k, v in self.model_vars.items() if v.get()]
        n_models = len(selected_models)
        split = self.split_var.get() if self.split_var else "—"

        text = (
            f"Task type     : {task.capitalize()}\n"
            f"Target column : {target}{' (diabaikan untuk anomali)' if is_anomaly else ''}\n"
            f"Features      : {n_feat} column(s) selected\n"
            f"Models        : {n_models} model(s) selected\n"
            f"Split ratio   : {split}\n"
        )
        self._summary_box.delete("1.0", "end")
        self._summary_box.insert("1.0", text)

    # ─────────────────────────────────────────────────────────────────────────
    #  TRAINING
    # ─────────────────────────────────────────────────────────────────────────
    def confirm_and_start_training(self):
        is_anomaly = getattr(self.app, "inferred_task", "") == "anomaly"

        if not is_anomaly:
            target = self.target_var.get()
            if not target or target == self._placeholder:
                messagebox.showerror("Error", "Please select a target column!")
                return

        selected_models = [k for k, v in self.model_vars.items() if v.get()]
        if not selected_models:
            messagebox.showerror("Error", "Select at least one model!")
            return

        if is_anomaly:
            anomaly_models_list = get_available_models("anomaly")
            final_models_to_run = [
                m for m in selected_models if m in anomaly_models_list
            ]
            if not final_models_to_run:
                messagebox.showerror("Error", "No trainable anomaly model selected!")
                return
        else:
            supervised_models = set(get_available_models("classification")) | set(
                get_available_models("regression")
            )
            final_models_to_run = [m for m in selected_models if m in supervised_models]
            if not final_models_to_run:
                messagebox.showerror(
                    "Error", "No trainable classification/regression model selected!"
                )
                return

        selected_features = [col for col, var in self.feature_vars.items() if var.get()]
        if not selected_features and not is_anomaly:
            messagebox.showerror("Error", "Please select at least one feature!")
            return

        self.app.target_column = self.target_var.get() if not is_anomaly else None
        self.app.selected_models = final_models_to_run
        self.app.selected_features = selected_features
        self.app.train_test_split = self.split_var.get()
        self.start_training()

    def start_training(self):
        if self.app.is_training:
            messagebox.showwarning("In Progress", "Training is already in progress.")
            return
        if not getattr(self.app, "inferred_task", None):
            messagebox.showerror("Error", "Please configure training first.")
            return
        self.app.cancel_training = False
        self.app.run_training()
        self._go_to_main_step(self.MAIN_STEP_TRAIN)

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 4 — RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    def show_results_screen(self):
        self.clear_content()

        results = self.app.training_results

        # ── PERBAIKAN UTAMA: all_trained_models tidak wajib ada di sini ───────
        # Kita hanya butuh results (dari JSON) untuk display metrik.
        # all_trained_models hanya dibutuhkan saat user mau konfirmasi model
        # (untuk Try Model / Export). Itu di-handle di _confirm_model_selection.
        all_trained_models = getattr(self.app, "all_trained_models", {})

        is_anomaly = getattr(self.app, "inferred_task", "") == "anomaly"

        if not results:
            ctk.CTkLabel(
                self.main_area, text="❌ No results to display.", font=("Arial", 20)
            ).grid(row=0, column=0, pady=50)
            return

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
            text="📊  Training Results",
            font=ctk.CTkFont(family="Arial", size=28, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            header_frame,
            text="❓ Apa Arti Metrik Ini?",
            width=180,
            height=36,
            fg_color="#3d4045",
            hover_color="#4a4d52",
            command=self.show_metrics_explanation_popup,
        ).grid(row=0, column=1, sticky="e")

        content_frame = ctk.CTkFrame(self.main_area, fg_color="transparent")
        content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 10))
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=0)
        content_frame.grid_columnconfigure(1, weight=1)

        model_list_frame = ctk.CTkScrollableFrame(
            content_frame,
            width=240,
            label_text="Pilih Model",
            label_font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
        )
        model_list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        model_list_frame.grid_columnconfigure(0, weight=1)

        self.model_buttons = {}
        self.selected_model_name_for_display = None

        for model_name in results.keys():
            btn = ctk.CTkButton(
                model_list_frame,
                text=model_name,
                command=lambda m=model_name: self._on_model_button_click(m),
                font=ctk.CTkFont(family="Arial", size=15, weight="bold"),
                fg_color="#2B2D30",
                hover_color="#3A3D40",
                anchor="w",
                height=48,
                corner_radius=8,
            )
            btn.pack(fill="x", pady=5, padx=5)
            self.model_buttons[model_name] = btn

        self.metrics_display_frame = ctk.CTkFrame(
            content_frame, fg_color="#2B2D30", corner_radius=12
        )
        self.metrics_display_frame.grid(row=0, column=1, sticky="nsew")
        self.metrics_display_frame.grid_columnconfigure(0, weight=1)
        self.metrics_display_frame.grid_rowconfigure(0, weight=0)
        self.metrics_display_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self.metrics_display_frame,
            text="Detail Model",
            font=ctk.CTkFont(family="Arial", size=22, weight="bold"),
        ).grid(row=0, column=0, pady=(20, 10))

        self.metrics_content_label = ctk.CTkTextbox(
            self.metrics_display_frame,
            corner_radius=12,
            font=("Consolas", 14),
            wrap="none",
            border_spacing=12,
        )
        self.metrics_content_label.grid(
            row=1, column=0, padx=20, pady=(10, 20), sticky="nsew"
        )
        self.metrics_content_label.insert(
            "1.0", "Pilih model dari daftar di samping untuk melihat detail metrik."
        )
        self.metrics_content_label.configure(state="disabled")

        self.nav_area.grid_columnconfigure(0, weight=0)
        self.nav_area.grid_columnconfigure(1, weight=1)
        self.nav_area.grid_columnconfigure(2, weight=0)

        ctk.CTkButton(
            self.nav_area,
            text="← Back: Configure",
            width=180,
            height=44,
            fg_color="#3d4045",
            hover_color="#4a4d52",
            font=ctk.CTkFont(family="Arial", size=15),
            command=lambda: self._go_to_main_step(self.MAIN_STEP_CONFIGURE),
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))

        mid_frame = ctk.CTkFrame(self.nav_area, fg_color="transparent")
        mid_frame.grid(row=0, column=1, sticky="ew")
        mid_frame.grid_columnconfigure(0, weight=1)

        self.selected_model_display_label = ctk.CTkLabel(
            mid_frame,
            text="Model Terpilih: Belum Ada",
            font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
            text_color="#3498db",
        )
        self.selected_model_display_label.grid(row=0, column=0, pady=(0, 4))

        # ── Tombol konfirmasi: hanya enable jika model sudah di-load ke memori ─
        model_loaded_in_memory = bool(all_trained_models)
        self.select_model_btn = ctk.CTkButton(
            mid_frame,
            text="✅ Konfirmasi Model Terpilih",
            width=240,
            height=38,
            font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
            fg_color="#27ae60",
            hover_color="#219a52",
            command=self._confirm_model_selection,
            state="normal" if model_loaded_in_memory else "disabled",
        )
        self.select_model_btn.grid(row=1, column=0)

        # ── Info jika model belum di-load ─────────────────────────────────────
        if not model_loaded_in_memory:
            ctk.CTkLabel(
                mid_frame,
                text="⚠️ Model sedang dimuat di background...",
                font=ctk.CTkFont(family="Arial", size=11),
                text_color="#f39c12",
            ).grid(row=2, column=0, pady=(2, 0))

        # ── "Next: Try Model" button — enable jika sudah ada selected_best_model ─
        can_proceed = bool(
            self.app.selected_best_model_name
            and getattr(self.app, "selected_best_model", None)
        )
        self.next_button_results = ctk.CTkButton(
            self.nav_area,
            text="Next: Try Model →",
            width=200,
            height=44,
            fg_color="#3498db",
            hover_color="#2980b9",
            font=ctk.CTkFont(family="Arial", size=15, weight="bold"),
            command=lambda: self._go_to_main_step(self.MAIN_STEP_TRY_MODEL),
            state="normal" if can_proceed else "disabled",
        )
        self.next_button_results.grid(row=0, column=2, sticky="e", padx=(8, 0))

        # ── Auto-klik model yang sudah terpilih ──────────────────────────────
        if (
            self.app.selected_best_model_name
            and self.app.selected_best_model_name in self.model_buttons
        ):
            self._on_model_button_click(self.app.selected_best_model_name)
            # Update label meski belum konfirmasi ulang
            self.selected_model_display_label.configure(
                text=f"Model Terpilih: {self.app.selected_best_model_name}"
            )

    def _on_model_button_click(self, model_name):
        if (
            self.selected_model_name_for_display
            and self.selected_model_name_for_display in self.model_buttons
        ):
            self.model_buttons[self.selected_model_name_for_display].configure(
                fg_color="#2B2D30", text_color="white"
            )

        if model_name in self.model_buttons:
            self.model_buttons[model_name].configure(
                fg_color="#3498db", text_color="white"
            )
            self.selected_model_name_for_display = model_name
            self.selected_model_display_label.configure(
                text=f"Model Terpilih: {model_name}"
            )

            if (
                hasattr(self, "select_model_btn")
                and self.select_model_btn.winfo_exists()
            ):
                # Enable konfirmasi hanya jika model sudah ada di memori
                all_trained = getattr(self.app, "all_trained_models", {})
                self.select_model_btn.configure(
                    state="normal" if all_trained else "disabled"
                )

            metrics = self.app.training_results.get(model_name, {})
            evaluation_detail = self.app.model_evaluation_details.get(model_name, {})
            summary = evaluation_detail.get("summary", {})

            # Fallback ke outlier_summary
            if not summary and hasattr(self.app, "outlier_summary"):
                summary = self.app.outlier_summary.get(model_name, {})

            # Fallback ke training_results langsung (untuk data yang sudah tersimpan di JSON)
            if not summary and metrics:
                summary = metrics

            scored_df = evaluation_detail.get("scored_dataset", pd.DataFrame())

            # Coba ambil dari scored_datasets jika evaluation_detail tidak punya
            if scored_df.empty and hasattr(self.app, "scored_datasets"):
                scored_df = self.app.scored_datasets.get(model_name, pd.DataFrame())

            metrics_text = f"Model: {model_name}\n\n"

            if "error" in metrics:
                metrics_text += f"Error: {metrics['error']}\n"
            else:
                if getattr(self.app, "inferred_task", "") == "anomaly":
                    src = summary if summary else metrics
                    metrics_text += "========== MODEL RESULT SUMMARY ==========\n\n"
                    metrics_text += f"Total Rows: {src.get('total_rows', 0):,}\n"
                    metrics_text += f"Total Outliers Detected: {src.get('total_outliers_detected', 0):,}\n"
                    metrics_text += (
                        f"Outlier Percentage: {src.get('outlier_percentage', 0)}%\n"
                    )
                    metrics_text += f"Lowest Outlier Score: {src.get('lowest_outlier_score', 0):.4f}\n"
                    metrics_text += f"Highest Outlier Score: {src.get('highest_outlier_score', 0):.4f}\n"
                    metrics_text += f"Average Outlier Score: {src.get('average_outlier_score', 0):.4f}\n"
                    metrics_text += (
                        f"Training Duration: {src.get('training_duration', 0)} sec\n"
                    )
                    metrics_text += f"Model Name: {src.get('model_name', '-')}\n"
                    metrics_text += (
                        f"Model Config:\n{src.get('model_configuration', {})}\n"
                    )

                    metrics_text += "\n========== INSIGHT SUMMARY ==========\n\n"
                    insights = src.get("insights", [])
                    if insights:
                        for item in insights:
                            metrics_text += f"• {item}\n"
                    else:
                        metrics_text += "• No major unusual pattern detected.\n"

                    metrics_text += "\n========== RECOMMENDATIONS ==========\n\n"
                    recommendations = src.get("recommendations", [])
                    if recommendations:
                        for item in recommendations:
                            metrics_text += f"• {item}\n"
                    else:
                        metrics_text += "• Model appears stable.\n"

                    metrics_text += "\n========== TOP OUTLIER INSPECTION ==========\n\n"
                    if not scored_df.empty:
                        top_outliers = scored_df.nsmallest(20, "outlier_score")
                        wanted_cols = [
                            "outlier_rank",
                            "outlier_score",
                            "Gross_Transaction_Amount",
                            "Annual_Income",
                            "txn_income_ratio",
                            "txn_vs_personal_avg",
                            "days_since_last_txn",
                            "portfolio_impact",
                            "Risk_Profile",
                            "Fund_Risk_Level",
                            "Transaction_Type",
                        ]
                        available_cols = [
                            c for c in wanted_cols if c in top_outliers.columns
                        ]
                        metrics_text += top_outliers[available_cols].to_string(
                            index=False
                        )
                    else:
                        metrics_text += "(Scored dataset dimuat di background, refresh halaman jika belum muncul)\n"
                else:
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            metrics_text += f"{metric}: {value:.4f}\n"
                        else:
                            metrics_text += f"{metric}: {value}\n"

            self.metrics_content_label.configure(state="normal")
            self.metrics_content_label.delete("1.0", "end")
            self.metrics_content_label.insert("1.0", metrics_text)
            self.metrics_content_label.configure(state="disabled")
        else:
            self.selected_model_name_for_display = None
            self.selected_model_display_label.configure(
                text="Model Terpilih: Belum Ada"
            )
            self.metrics_content_label.configure(state="normal")
            self.metrics_content_label.delete("1.0", "end")
            self.metrics_content_label.insert(
                "1.0", "Pilih model dari daftar di samping untuk melihat detail."
            )
            self.metrics_content_label.configure(state="disabled")
            if (
                hasattr(self, "select_model_btn")
                and self.select_model_btn.winfo_exists()
            ):
                self.select_model_btn.configure(state="disabled")

    def _confirm_model_selection(self):
        if not self.selected_model_name_for_display:
            messagebox.showwarning(
                "Pilih Model", "Klik model dari daftar terlebih dahulu."
            )
            return

        all_trained = getattr(self.app, "all_trained_models", {})
        if not all_trained:
            messagebox.showwarning(
                "Model Belum Siap",
                "Model masih dimuat di background. Tunggu sebentar lalu coba lagi.",
            )
            return

        model_pipeline = all_trained.get(self.selected_model_name_for_display)
        if model_pipeline is None:
            messagebox.showerror(
                "Error",
                f"Model '{self.selected_model_name_for_display}' tidak ditemukan di memori.\n"
                "Pastikan model sudah selesai dimuat.",
            )
            return

        self.app.set_selected_best_model(
            self.selected_model_name_for_display, model_pipeline
        )

        if (
            hasattr(self, "next_button_results")
            and self.next_button_results.winfo_exists()
        ):
            self.next_button_results.configure(state="normal")

        messagebox.showinfo(
            "Model Dikonfirmasi",
            f"Model '{self.selected_model_name_for_display}' berhasil dipilih!\n"
            "Anda bisa lanjut ke Try Model.",
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 5 — TRY MODEL
    # ─────────────────────────────────────────────────────────────────────────
    def show_try_model_screen(self):
        self.clear_content()
        self.sidebar_steps["try_model"].configure(state="normal")

        for i in range(20):
            self.main_area.grid_rowconfigure(i, weight=0)
        self.main_area.grid_rowconfigure(0, weight=0)
        self.main_area.grid_rowconfigure(1, weight=0)
        self.main_area.grid_rowconfigure(2, weight=0)
        self.main_area.grid_rowconfigure(3, weight=1)
        self.main_area.grid_rowconfigure(4, weight=0)
        self.main_area.grid_rowconfigure(5, weight=0)
        self.main_area.grid_columnconfigure(0, weight=1)

        task = getattr(self.app, "inferred_task", "classification")

        ctk.CTkLabel(
            self.main_area,
            text="🧪  Test Your Model",
            font=ctk.CTkFont(family="Arial", size=34, weight="bold"),
        ).grid(row=0, column=0, pady=(30, 8), padx=20, sticky="w")

        subtitle = (
            "Masukkan nilai fitur untuk memeriksa apakah data ini anomali."
            if task == "anomaly"
            else "Masukkan nilai fitur di bawah untuk mendapatkan prediksi."
        )
        ctk.CTkLabel(
            self.main_area,
            text=subtitle,
            font=ctk.CTkFont(family="Arial", size=16),
            text_color="gray60",
        ).grid(row=1, column=0, pady=(0, 16), padx=20, sticky="w")

        if (
            not hasattr(self.app, "selected_best_model")
            or not self.app.selected_best_model
        ):
            ctk.CTkLabel(
                self.main_area,
                text="⚠️  Belum ada model yang dipilih. Kembali ke Results dan konfirmasi model terlebih dahulu.",
                font=ctk.CTkFont(family="Arial", size=15),
                text_color="#e74c3c",
                wraplength=700,
                justify="left",
            ).grid(row=2, column=0, pady=40, padx=20)
            self._nav_buttons(
                show_back=True,
                back_label="← Back: Results",
                back_cmd=lambda: self._go_to_main_step(self.MAIN_STEP_RESULTS),
                show_next=False,
            )
            return

        input_frame = ctk.CTkScrollableFrame(
            self.main_area,
            label_text=f"Feature Inputs  —  Model: {self.app.selected_best_model_name}",
        )
        input_frame.grid(row=3, column=0, pady=10, padx=20, sticky="nsew")
        input_frame.grid_columnconfigure(0, weight=1)

        self.input_widgets = {}
        col_types = getattr(self.app, "feature_column_types", {})

        for i, feature in enumerate(self.app.selected_features):
            col_type = col_types.get(feature)
            if col_type is None:
                dtype = self.app.df[feature].dtype
                col_type = (
                    "numeric" if pd.api.types.is_numeric_dtype(dtype) else "categorical"
                )

            row_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
            row_frame.grid(row=i, column=0, sticky="ew", pady=5, padx=10)
            row_frame.grid_columnconfigure(0, minsize=220)
            row_frame.grid_columnconfigure(1, weight=1)

            type_hint = {
                "numeric": "angka",
                "datetime": "tanggal",
                "categorical": "kategori",
            }.get(col_type, "")
            ctk.CTkLabel(
                row_frame,
                text=f"{feature}  ({type_hint}):",
                anchor="w",
                font=ctk.CTkFont(size=14, weight="bold"),
            ).grid(row=0, column=0, sticky="w", padx=(0, 10))

            if col_type == "numeric":
                entry = ctk.CTkEntry(
                    row_frame,
                    font=ctk.CTkFont(size=14),
                    placeholder_text="contoh: 1234.56",
                )
                entry.grid(row=0, column=1, sticky="ew")
                self.input_widgets[feature] = ("numeric", entry)
            elif col_type == "datetime":
                entry = ctk.CTkEntry(
                    row_frame,
                    font=ctk.CTkFont(size=14),
                    placeholder_text="contoh: 2019-11-11",
                )
                entry.grid(row=0, column=1, sticky="ew")
                self.input_widgets[feature] = ("datetime", entry)
            else:
                unique_values = [
                    str(val) for val in self.app.df[feature].dropna().unique()
                ]
                if not unique_values:
                    unique_values = ["N/A"]
                option_menu = ctk.CTkOptionMenu(
                    row_frame, values=unique_values, font=ctk.CTkFont(size=14)
                )
                option_menu.set(unique_values[0])
                option_menu.grid(row=0, column=1, sticky="ew")
                self.input_widgets[feature] = ("categorical", option_menu)

        self.prediction_label = ctk.CTkLabel(
            self.main_area,
            text="Prediction: Awaiting input...",
            font=ctk.CTkFont(family="Arial", size=20, weight="bold"),
            text_color="#3498db",
        )
        self.prediction_label.grid(row=4, column=0, pady=(10, 4))

        self.anomaly_threshold_var = ctk.StringVar(
            value=str(getattr(self.app, "anomaly_threshold", 0.0))
        )
        if task == "anomaly":
            threshold_frame = ctk.CTkFrame(self.main_area, fg_color="transparent")
            threshold_frame.grid(row=5, column=0, pady=(0, 10), padx=20, sticky="ew")
            threshold_frame.grid_columnconfigure(0, weight=0)
            threshold_frame.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(
                threshold_frame,
                text="Anomaly Threshold (decision_function):",
                font=ctk.CTkFont(size=14, weight="bold"),
                anchor="w",
            ).grid(row=0, column=0, sticky="w", padx=(0, 10))
            threshold_entry = ctk.CTkEntry(
                threshold_frame,
                textvariable=self.anomaly_threshold_var,
                font=ctk.CTkFont(size=14),
                placeholder_text="contoh: -0.1",
            )
            threshold_entry.grid(row=0, column=1, sticky="ew")
            self.app.anomaly_threshold = float(self.anomaly_threshold_var.get())

        ctk.CTkButton(
            self.main_area,
            text="🔮  Predict",
            width=220,
            height=52,
            font=ctk.CTkFont(family="Arial", size=18, weight="bold"),
            fg_color="#27ae60",
            hover_color="#219a52",
            command=self._make_prediction,
        ).grid(row=6 if task == "anomaly" else 5, column=0, pady=(0, 10))

        self._nav_buttons(
            show_back=True,
            back_label="← Back: Results",
            back_cmd=lambda: self._go_to_main_step(self.MAIN_STEP_RESULTS),
            next_label="Next: Export Model →",
            next_cmd=lambda: self._go_to_main_step(self.MAIN_STEP_EXPORT_MODEL),
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  DYNAMIC ANOMALY EXPLAINER
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_anomaly_explanation(self, input_df, anomaly_score, threshold):
        score_stats = getattr(self.app, "score_stats", None)
        if not score_stats:
            return {
                "severity": "UNKNOWN",
                "why_flagged": ["Score statistics unavailable"],
            }

        p0_1 = score_stats["p0_1"]
        p1 = score_stats["p1"]
        p3 = score_stats["p3"]

        if anomaly_score <= p0_1:
            severity = "CRITICAL"
        elif anomaly_score <= p1:
            severity = "HIGH"
        elif anomaly_score <= p3:
            severity = "MEDIUM"
        elif anomaly_score < threshold:
            severity = "LOW"
        else:
            severity = "NORMAL"

        reasons = []
        feature_stats = getattr(self.app, "feature_stats", None)
        if not feature_stats:
            return {
                "severity": severity,
                "why_flagged": ["Feature statistics unavailable"],
            }

        row = input_df.iloc[0]
        extreme_count = 0
        moderate_count = 0

        for feature, value in row.items():
            if feature not in feature_stats:
                continue
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            stats = feature_stats[feature]
            mean = stats["mean"]
            std = stats["std"]
            if std <= 0:
                continue
            zscore = abs((value - mean) / std)
            if zscore >= 3:
                reasons.append(f"{feature} extremely abnormal")
                extreme_count += 1
            elif zscore >= 2:
                reasons.append(f"{feature} unusually deviated")
                moderate_count += 1

        if extreme_count >= 6:
            severity = "CRITICAL"
        elif extreme_count >= 4:
            severity = "HIGH"
        elif extreme_count >= 2 and severity == "LOW":
            severity = "MEDIUM"

        if not reasons:
            if anomaly_score < threshold:
                reasons.append("General behavioral anomaly")
            else:
                reasons.append("Behavior within normal range")

        return {
            "severity": severity,
            "why_flagged": reasons,
            "extreme_feature_count": extreme_count,
            "moderate_feature_count": moderate_count,
        }

    def _make_prediction(self):
        if (
            not hasattr(self.app, "selected_best_model")
            or not self.app.selected_best_model
        ):
            messagebox.showerror("Error", "No selected model available for prediction.")
            return

        input_data = {}
        try:
            for feature, (col_type, widget) in self.input_widgets.items():
                value = widget.get()
                if col_type == "numeric":
                    if value.strip() == "":
                        messagebox.showerror(
                            "Input Error",
                            f"Kolom numerik '{feature}' tidak boleh kosong.",
                        )
                        return
                    try:
                        input_data[feature] = [float(value.replace(",", "."))]
                    except ValueError:
                        messagebox.showerror(
                            "Input Error",
                            f"Input tidak valid untuk '{feature}': '{value}'.",
                        )
                        return
                elif col_type == "datetime":
                    input_data[feature] = [
                        None if value.strip() == "" else value.strip()
                    ]
                else:
                    input_data[feature] = [value]
        except Exception as e:
            messagebox.showerror("Input Error", f"Error memproses input: {e}")
            return

        try:
            input_df = pd.DataFrame(input_data, columns=self.app.selected_features)
            task = self.app.inferred_task

            if task == "anomaly":
                try:
                    threshold_str = self.anomaly_threshold_var.get()
                    if not threshold_str:
                        messagebox.showerror(
                            "Input Error", "Anomaly Threshold tidak boleh kosong."
                        )
                        return
                    anomaly_threshold = float(threshold_str)
                    self.app.anomaly_threshold = anomaly_threshold
                    anomaly_score = self.app.selected_best_model.decision_function(
                        input_df
                    )[0]

                    explanation = self._generate_anomaly_explanation(
                        input_df, anomaly_score, anomaly_threshold
                    )

                    if anomaly_score < anomaly_threshold:
                        color = "#E74C3C"
                        result_text = (
                            f"Result: Anomaly Detected 🚨\n\n"
                            f"Severity: {explanation['severity']}\n"
                            f"Score: {anomaly_score:.4f}\n\n"
                            f"Why Flagged:\n"
                            f"- " + "\n- ".join(explanation["why_flagged"])
                        )
                    else:
                        color = "#2ECC71"
                        result_text = (
                            f"Result: Normal ✅\n\n"
                            f"Severity: {explanation['severity']}\n"
                            f"Score: {anomaly_score:.4f}\n\n"
                            f"Status:\n"
                            f"- " + "\n- ".join(explanation["why_flagged"])
                        )
                    self.prediction_label.configure(text=result_text, text_color=color)

                except ValueError:
                    messagebox.showerror(
                        "Input Error", f"Threshold tidak valid: '{threshold_str}'."
                    )
                    return
                except Exception as e:
                    messagebox.showerror(
                        "Prediction Error", f"Error anomaly prediction: {e}"
                    )
                    return
            elif task == "classification":
                prediction_val = self.app.selected_best_model.predict(input_df)[0]
                display_prediction = prediction_val
                if self.app.label_encoder:
                    display_prediction = self.app.label_encoder.inverse_transform(
                        [int(prediction_val)]
                    )[0]
                prob_text = ""
                if hasattr(self.app.selected_best_model, "predict_proba"):
                    probabilities = self.app.selected_best_model.predict_proba(
                        input_df
                    )[0]
                    class_labels = (
                        self.app.label_encoder.classes_
                        if self.app.label_encoder
                        else self.app.selected_best_model.classes_
                    )
                    prob_parts = [
                        f"{lbl}: {p:.1%}" for lbl, p in zip(class_labels, probabilities)
                    ]
                    prob_text = f"\nProbabilities: {', '.join(prob_parts)}"
                self.prediction_label.configure(
                    text=f"Prediction: {display_prediction}{prob_text}",
                    text_color="#3498db",
                )
            elif task == "regression":
                prediction_val = self.app.selected_best_model.predict(input_df)[0]
                self.prediction_label.configure(
                    text=f"Prediction: {prediction_val:.2f}", text_color="#3498db"
                )
        except Exception as e:
            messagebox.showerror(
                "Prediction Error", f"An error occurred during prediction: {e}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 6 — EXPORT MODEL
    # ─────────────────────────────────────────────────────────────────────────
    def show_export_model_screen(self):
        self.clear_content()
        self.sidebar_steps["export_model"].configure(state="normal")

        for i in range(20):
            self.main_area.grid_rowconfigure(i, weight=0)
        self.main_area.grid_rowconfigure(4, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.main_area,
            text="📦  Export Your Model",
            font=ctk.CTkFont(family="Arial", size=34, weight="bold"),
        ).grid(row=0, column=0, pady=(30, 8), padx=20, sticky="w")

        if (
            not hasattr(self.app, "selected_best_model_path")
            or not self.app.selected_best_model_path
        ):
            ctk.CTkLabel(
                self.main_area,
                text="⚠️  Belum ada model yang dipilih. Kembali ke Results dan konfirmasi model terlebih dahulu.",
                font=ctk.CTkFont(family="Arial", size=15),
                text_color="#e74c3c",
                wraplength=700,
                justify="left",
            ).grid(row=1, column=0, pady=40, padx=20)
            self._nav_buttons(
                show_back=True,
                back_label="← Back: Try Model",
                back_cmd=lambda: self._go_to_main_step(self.MAIN_STEP_TRY_MODEL),
                show_next=False,
            )
            return

        ctk.CTkLabel(
            self.main_area,
            text="Model tersimpan di:",
            font=ctk.CTkFont(family="Arial", size=15),
            text_color="gray60",
        ).grid(row=1, column=0, pady=(0, 4), padx=20, sticky="w")

        ctk.CTkEntry(
            self.main_area,
            width=700,
            height=40,
            font=ctk.CTkFont(family="Arial", size=14),
            textvariable=ctk.StringVar(value=self.app.selected_best_model_path),
            state="readonly",
        ).grid(row=2, column=0, pady=(0, 20), padx=20, sticky="w")

        options_frame = ctk.CTkFrame(self.main_area, fg_color="transparent")
        options_frame.grid(row=3, column=0, pady=10, padx=20, sticky="w")

        ctk.CTkButton(
            options_frame,
            text="💾  Copy Model File (.joblib)",
            width=300,
            height=52,
            font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
            fg_color="#3498db",
            hover_color="#2980b9",
            command=self._copy_model_file,
        ).grid(row=0, column=0, padx=(0, 16))

        ctk.CTkButton(
            options_frame,
            text="🐍  Generate Python Example",
            width=300,
            height=52,
            font=ctk.CTkFont(family="Arial", size=16, weight="bold"),
            fg_color="#27ae60",
            hover_color="#219a52",
            command=self._generate_python_example,
        ).grid(row=0, column=1)

        ctk.CTkLabel(
            self.main_area,
            text="Format lain (ONNX, TensorFlow Lite) segera hadir!",
            font=ctk.CTkFont(family="Arial", size=13),
            text_color="gray50",
        ).grid(row=4, column=0, pady=(20, 0), padx=20, sticky="w")

        self._nav_buttons(
            show_back=True,
            back_label="← Back: Try Model",
            back_cmd=lambda: self._go_to_main_step(self.MAIN_STEP_TRY_MODEL),
            show_next=False,
        )

    def _copy_model_file(self):
        if (
            not hasattr(self.app, "selected_best_model_path")
            or not self.app.selected_best_model_path
        ):
            messagebox.showerror("Error", "No selected model path available.")
            return
        destination_path = filedialog.asksaveasfilename(
            title="Save Model As",
            initialfile=os.path.basename(self.app.selected_best_model_path),
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")],
        )
        if destination_path:
            try:
                import shutil

                shutil.copy(self.app.selected_best_model_path, destination_path)
                messagebox.showinfo("Success", f"Model copied to:\n{destination_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy model file: {e}")

    def _generate_python_example(self):
        if (
            not hasattr(self.app, "selected_best_model_path")
            or not self.app.selected_best_model_path
        ):
            messagebox.showerror("Error", "No selected model path available.")
            return
        if not hasattr(self.app, "selected_features") or not self.app.selected_features:
            messagebox.showerror("Error", "Selected features not available.")
            return
        if not hasattr(self.app, "inferred_task") or not self.app.inferred_task:
            messagebox.showerror("Error", "Inferred task type not available.")
            return

        lines = [
            "import joblib",
            "import pandas as pd",
            "",
            f'model_path = r"{self.app.selected_best_model_path}"',
            "loaded_data = joblib.load(model_path)",
            "pipeline = loaded_data['pipeline']",
            "label_encoder = loaded_data.get('label_encoder')",
            "",
            f"features = {self.app.selected_features}",
            "",
            "input_data = {",
        ]
        for feature in self.app.selected_features:
            if pd.api.types.is_numeric_dtype(self.app.df[feature].dtype):
                lines.append(f"    '{feature}': [0.0],")
            elif self.app.df[feature].dtype in ("object", "category"):
                first_val = (
                    str(self.app.df[feature].dropna().iloc[0])
                    if not self.app.df[feature].dropna().empty
                    else "example_category"
                )
                lines.append(f"    '{feature}': ['{first_val}'],")
            else:
                lines.append(f"    '{feature}': ['example_value'],")

        task = self.app.inferred_task
        lines += [
            "}",
            "",
            "input_df = pd.DataFrame(input_data, columns=features)",
            "prediction = pipeline.predict(input_df)[0]",
            "print(f'Raw Prediction: {prediction}')",
            "",
            f"if '{task}' == 'anomaly':",
            "    print('Result: Anomaly Detected' if prediction == -1 else 'Result: Normal')",
            f"elif '{task}' == 'classification' and label_encoder is not None:",
            "    decoded = label_encoder.inverse_transform([int(prediction)])[0]",
            "    print(f'Decoded Prediction: {decoded}')",
            f"elif '{task}' == 'regression':",
            "    print(f'Regression Prediction: {prediction:.2f}')",
        ]
        example_code = "\n".join(lines)

        code_window = ctk.CTkToplevel(self.app)
        code_window.title("Python Usage Example")
        code_window.geometry("800x600")
        code_window.grab_set()

        textbox = ctk.CTkTextbox(
            code_window, wrap="word", font=("Consolas", 12), border_spacing=12
        )
        textbox.pack(fill="both", expand=True, padx=10, pady=10)
        textbox.insert("1.0", example_code)
        textbox.configure(state="disabled")

        ctk.CTkButton(code_window, text="Tutup", command=code_window.destroy).pack(
            pady=10
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  METRICS POPUP
    # ─────────────────────────────────────────────────────────────────────────
    def show_metrics_explanation_popup(self):
        popup = ctk.CTkToplevel(self.app)
        popup.title("Penjelasan Metrik Evaluasi")
        popup.geometry("800x750")
        popup.grab_set()
        popup.resizable(False, False)

        scrollable_frame = ctk.CTkScrollableFrame(
            popup, label_text="Penjelasan 'Nilai Rapor' Model"
        )
        scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)

        for section_title, body_text in [
            (
                "Untuk Deteksi Anomali (menemukan data aneh)",
                (
                    "• Outliers Detected (Anomali Terdeteksi): Jumlah data yang dianggap 'aneh' "
                    "atau 'berbeda' oleh model dari keseluruhan data uji.\n\n"
                    "• Outlier Percentage (Persentase Anomali): Persentase data anomali dari "
                    "total data uji. Tidak ada nilai 'benar' atau 'salah' — ini tergantung "
                    "pada konteks bisnis Anda."
                ),
            ),
            (
                "Untuk Klasifikasi (memprediksi kategori)",
                (
                    "• Accuracy (Akurasi): Persentase tebakan benar dari total keseluruhan.\n\n"
                    "• Precision (Presisi): Dari semua yang ditebak 'Positif', berapa persen "
                    "yang benar. Berguna untuk menghindari 'alarm palsu'.\n\n"
                    "• Recall (Daya Ingat): Dari semua kasus 'Positif' asli, berapa persen "
                    "berhasil ditemukan.\n\n"
                    "• F1-Score: Nilai gabungan Precision dan Recall. Metrik terbaik untuk "
                    "data yang tidak seimbang."
                ),
            ),
            (
                "Untuk Regresi (memprediksi angka)",
                (
                    "• MSE (Mean Squared Error): Rata-rata kuadrat dari selisih prediksi dan "
                    "nilai asli. Semakin KECIL, semakin bagus.\n\n"
                    "• R2 Score: Seberapa baik model mengikuti variasi data asli (skala 0–1). "
                    "Semakin MENDEKATI 1, semakin bagus."
                ),
            ),
        ]:
            ctk.CTkLabel(
                scrollable_frame,
                text=section_title,
                font=ctk.CTkFont(size=18, weight="bold", underline=True),
            ).pack(anchor="w", padx=10, pady=(20, 5))
            ctk.CTkLabel(
                scrollable_frame,
                text=body_text,
                justify="left",
                wraplength=750,
                font=ctk.CTkFont(size=14),
            ).pack(anchor="w", padx=15, pady=5)

        ctk.CTkButton(popup, text="Tutup", command=popup.destroy, width=100).pack(
            pady=20
        )
