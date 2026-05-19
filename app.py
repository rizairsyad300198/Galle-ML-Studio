# ================================
# app.py  (FIXED)
# ================================

import customtkinter as ctk
import pandas as pd
import os
import shutil
import threading
import joblib
from PIL import Image

from tkinter import filedialog, messagebox

from core.project_manager import ProjectManager
from core.trainer import (
    MLTrainer,
    get_column_types,
)

from ui.main import StartScreen, WorkspaceScreen
from helper.helper import _extract_feature_statistics

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class GalleMLStudio(ctk.CTk):

    def __init__(self):

        super().__init__()
        self._setup_window_icon()
        self.title("Galle ML Studio")

        # Set geometry awal sebagai fallback
        self.geometry("1600x950")

        # Maximize window setelah event loop siap
        # (harus via after() agar bekerja di Windows — state("zoomed") di __init__ sering diabaikan)
        self._show_splash()

        def _maximize():
            try:
                self.state("zoomed")  # Windows: maximize dengan title bar
            except Exception:
                try:
                    self.attributes("-zoomed", True)  # Linux (GNOME/XFCE/KDE)
                except Exception:
                    # Fallback manual: set ke ukuran layar penuh
                    sw = self.winfo_screenwidth()
                    sh = self.winfo_screenheight()
                    self.geometry(f"{sw}x{sh}+0+0")

        self.after(10, _maximize)

    def _setup_state(self):
        # =====================================================
        # STATE
        # =====================================================

        self.project_manager = ProjectManager()

        self.current_project = None

        self.dataset_path = None

        self.df = None

        self.target_column = None

        self.selected_models = []

        self.selected_features = []

        self.train_test_split = "80/20"

        self.inferred_task = None

        self.training_results = None

        # all_trained_models: {model_name: pipeline_object}
        # Diisi saat training ATAU saat load project dari artifact files
        self.all_trained_models = {}

        self.label_encoder = None

        self.selected_best_model = None

        self.selected_best_model_name = None

        self.selected_best_model_path = None

        self.datetime_cols_for_prediction = []

        self.feature_column_types = {}

        self.user_model_params = {}

        self.is_training = False

        self.cancel_training = False

        # =====================================================
        # OUTLIER EVALUATION STATE
        # =====================================================

        self.model_evaluation_details = {}

        self.scored_datasets = {}

        self.training_durations = {}

        self.outlier_summary = {}

        self.outlier_insights = {}

        self.outlier_recommendations = {}

        # score_stats & feature_stats (untuk anomaly explainer di Try Model)
        self.score_stats = {}

        self.feature_stats = {}

        # =====================================================
        # SCREEN
        # =====================================================

        self.workspace_screen = None

        self.show_start_screen()

    def _setup_window_icon(self):
        """Set icon di taskbar dan title bar."""
        try:
            icon_path = os.path.join("assets", "logo.ico")
            if os.path.exists(icon_path):
                # ← via after() supaya tidak di-override CTk
                self.after(0, lambda: self.iconbitmap(icon_path))
            else:
                png_path = os.path.join("assets", "logo.png")
                if os.path.exists(png_path):
                    from PIL import ImageTk

                    img = ImageTk.PhotoImage(file=png_path)
                    self.after(0, lambda: self.iconphoto(True, img))
                    self._icon_ref = img
        except Exception as e:
            print(f"[icon] {e}")

    def _show_splash(self):
        """Tampilkan splash screen selama 2.5 detik, lalu lanjut ke app."""

        splash = ctk.CTkToplevel(self)
        splash.overrideredirect(True)  # hapus title bar
        splash.attributes("-topmost", True)

        # ── Ukuran & posisi splash ────────────────────────────────────────
        sw, sh = 420, 300
        self.update_idletasks()
        scr_w = self.winfo_screenwidth()
        scr_h = self.winfo_screenheight()
        x = scr_w // 2 - sw // 2
        y = scr_h // 2 - sh // 2
        splash.geometry(f"{sw}x{sh}+{x}+{y}")
        splash.configure(fg_color="#1a1a2e")

        # ── Rounded border effect ─────────────────────────────────────────
        container = ctk.CTkFrame(
            splash,
            corner_radius=20,
            fg_color="#1a1a2e",
            border_width=1,
            border_color="#3498db",
        )
        container.pack(fill="both", expand=True, padx=2, pady=2)

        # ── Logo image ────────────────────────────────────────────────────
        logo_path = os.path.join("assets", "splash.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join("assets", "logo.png")

        if os.path.exists(logo_path):
            logo_img = ctk.CTkImage(
                light_image=Image.open(logo_path),
                dark_image=Image.open(logo_path),
                size=(100, 100),
            )
            ctk.CTkLabel(
                container,
                image=logo_img,
                text="",
            ).pack(pady=(40, 8))
        else:
            ctk.CTkLabel(
                container,
                text="G",
                font=ctk.CTkFont(family="Arial", size=64, weight="bold"),
                text_color="#3498db",
            ).pack(pady=(40, 8))

        # ── Teks ─────────────────────────────────────────────────────────
        ctk.CTkLabel(
            container,
            text="Galle ML Studio",
            font=ctk.CTkFont(family="Arial", size=22, weight="bold"),
            text_color="white",
        ).pack()

        ctk.CTkLabel(
            container,
            text="Professional Machine Learning Workspace",
            font=ctk.CTkFont(family="Arial", size=12),
            text_color="gray60",
        ).pack(pady=(4, 0))

        # ── Loading bar animasi ───────────────────────────────────────────
        bar_track = ctk.CTkFrame(
            container,
            height=4,
            fg_color="#2d2d2d",
            corner_radius=2,
        )
        bar_track.pack(fill="x", padx=40, pady=(20, 0))

        bar_fill = ctk.CTkFrame(
            bar_track,
            height=4,
            width=0,
            fg_color="#3498db",
            corner_radius=2,
        )
        bar_fill.place(x=0, y=0)

        ctk.CTkLabel(
            container,
            text="v1.0.0",
            font=ctk.CTkFont(family="Arial", size=10),
            text_color="gray40",
        ).pack(pady=(8, 0))

        # ── Animasi loading bar ───────────────────────────────────────────
        duration_ms = 2500
        steps = 50
        interval = duration_ms // steps

        def _animate(step=0):
            if not splash.winfo_exists():
                return
            try:
                total_w = bar_track.winfo_width() or 340
                new_w = int((step / steps) * total_w)
                bar_fill.configure(width=new_w)
                if step < steps:
                    splash.after(interval, lambda: _animate(step + 1))
                else:
                    splash.after(200, _close_splash)
            except Exception:
                _close_splash()

        def _close_splash():
            try:
                splash.destroy()
            except Exception:
                pass
            self._init_app()

        splash.after(100, lambda: _animate(0))

        # Sembunyikan main window selama splash
        self.withdraw()

    def _init_app(self):
        """Dipanggil setelah splash selesai — tampilkan main window."""
        self.deiconify()
        self._setup_window_icon()
        self._setup_state()
        self.show_start_screen()

    # =========================================================
    # SCREEN
    # =========================================================

    def clear_window(self):

        for widget in self.winfo_children():
            widget.destroy()

    def show_start_screen(self):

        self.clear_window()

        StartScreen(self).render()

    def show_workspace(self):

        self.clear_window()

        self.workspace_screen = WorkspaceScreen(self)

        self.workspace_screen.render()

    # =========================================================
    # PROJECT
    # =========================================================

    def create_project(self, project_name):

        try:

            self.project_manager.create_project(project_name)

            self.current_project = project_name

            self.show_workspace()

        except Exception as e:

            messagebox.showerror("Error", str(e))

    def open_project(self):

        projects = self.project_manager.list_projects()

        if not projects:
            messagebox.showinfo("Info", "No project found.")
            return

        popup = ctk.CTkToplevel(self)
        popup.title("Open Project")
        popup.resizable(False, False)
        popup.grab_set()
        popup.transient(self)
        popup.lift()

        popup_w = 520
        popup_h = min(640, max(360, 200 + len(projects) * 62))

        self.update_idletasks()
        x = self.winfo_rootx() + self.winfo_width() // 2 - popup_w // 2
        y = self.winfo_rooty() + self.winfo_height() // 2 - popup_h // 2
        popup.geometry(f"{popup_w}x{popup_h}+{x}+{y}")

        # ── State ─────────────────────────────────────────────────────────────
        selected_projects = set()  # untuk multi-select delete
        project_rows = {}  # {name: {"frame": ..., "check_var": ...}}

        # ── Header ───────────────────────────────────────────────────────────
        header = ctk.CTkFrame(popup, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(24, 0))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="📂  Open Project",
            font=ctk.CTkFont(family="Arial", size=24, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        # Mode toggle: Normal / Select
        mode_var = ctk.StringVar(value="normal")

        select_mode_btn = ctk.CTkButton(
            header,
            text="☑ Select",
            width=90,
            height=32,
            fg_color="#3d4045",
            hover_color="#4a4d52",
            font=ctk.CTkFont(family="Arial", size=13),
            command=lambda: _toggle_select_mode(),
        )
        select_mode_btn.grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            popup,
            text=f"{len(projects)} project tersedia",
            font=ctk.CTkFont(family="Arial", size=12),
            text_color="gray50",
        ).pack(anchor="w", padx=26, pady=(4, 8))

        sep = ctk.CTkFrame(popup, height=1, fg_color="#3d4045")
        sep.pack(fill="x", padx=24, pady=(0, 8))

        # ── Scrollable list ───────────────────────────────────────────────────
        scroll = ctk.CTkScrollableFrame(popup, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        def _build_project_rows():
            for w in scroll.winfo_children():
                w.destroy()
            project_rows.clear()

            current_projects = self.project_manager.list_projects()

            for project in current_projects:
                is_select_mode = mode_var.get() == "select"

                row_frame = ctk.CTkFrame(
                    scroll,
                    corner_radius=10,
                    fg_color="#2b2d30",
                    height=56,
                )
                row_frame.pack(fill="x", pady=4, padx=4)
                row_frame.pack_propagate(False)
                row_frame.grid_columnconfigure(1, weight=1)

                # ── Checkbox (hanya tampil di select mode) ─────────────────
                check_var = ctk.BooleanVar(value=False)

                check = ctk.CTkCheckBox(
                    row_frame,
                    text="",
                    variable=check_var,
                    width=28,
                    checkbox_width=20,
                    checkbox_height=20,
                    command=lambda p=project, v=check_var: _on_check(p, v),
                )
                if is_select_mode:
                    check.grid(row=0, column=0, padx=(12, 4), pady=12)
                else:
                    check.grid_remove()

                # ── Icon + nama project ───────────────────────────────────
                name_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
                name_frame.grid(row=0, column=1, sticky="ew", padx=(8, 0))

                ctk.CTkLabel(
                    name_frame,
                    text="📁",
                    font=ctk.CTkFont(size=18),
                ).pack(side="left", padx=(8, 6))

                ctk.CTkLabel(
                    name_frame,
                    text=project,
                    font=ctk.CTkFont(family="Arial", size=15, weight="bold"),
                    anchor="w",
                ).pack(side="left")

                # ── Tombol Open (hidden di select mode) ───────────────────
                open_btn = ctk.CTkButton(
                    row_frame,
                    text="Open →",
                    width=80,
                    height=34,
                    corner_radius=8,
                    fg_color="#3498db",
                    hover_color="#2980b9",
                    font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
                    command=lambda p=project: self._load_project(p, popup),
                )
                if not is_select_mode:
                    open_btn.grid(row=0, column=2, padx=(8, 12), pady=10)

                # ── Tombol Delete single (hidden di select mode) ──────────
                del_btn = ctk.CTkButton(
                    row_frame,
                    text="🗑",
                    width=36,
                    height=34,
                    corner_radius=8,
                    fg_color="#7f1d1d",
                    hover_color="#991b1b",
                    font=ctk.CTkFont(size=15),
                    command=lambda p=project: _delete_single(p),
                )
                if not is_select_mode:
                    del_btn.grid(row=0, column=3, padx=(0, 12), pady=10)

                project_rows[project] = {
                    "frame": row_frame,
                    "check_var": check_var,
                    "check": check,
                    "open_btn": open_btn,
                    "del_btn": del_btn,
                }

        def _toggle_select_mode():
            if mode_var.get() == "normal":
                mode_var.set("select")
                select_mode_btn.configure(
                    text="✖ Batal",
                    fg_color="#555",
                )
                selected_projects.clear()
                delete_bar.pack(fill="x", padx=16, pady=(0, 4), before=sep2)
            else:
                mode_var.set("normal")
                select_mode_btn.configure(
                    text="☑ Select",
                    fg_color="#3d4045",
                )
                selected_projects.clear()
                delete_bar.pack_forget()

            _build_project_rows()
            _update_delete_bar()

        def _on_check(project_name, var):
            if var.get():
                selected_projects.add(project_name)
            else:
                selected_projects.discard(project_name)
            _update_delete_bar()

        def _update_delete_bar():
            n = len(selected_projects)
            if n == 0:
                delete_selected_btn.configure(
                    state="disabled",
                    text="🗑  Hapus yang Dipilih",
                    fg_color="#7f1d1d",
                )
                select_all_btn.configure(text="☑ Pilih Semua")
            else:
                delete_selected_btn.configure(
                    state="normal",
                    text=f"🗑  Hapus {n} Project",
                    fg_color="#e74c3c",
                )
                all_projects = self.project_manager.list_projects()
                if len(selected_projects) == len(all_projects):
                    select_all_btn.configure(text="☐ Batal Semua")
                else:
                    select_all_btn.configure(text="☑ Pilih Semua")

        def _toggle_select_all():
            all_projects = self.project_manager.list_projects()
            if len(selected_projects) == len(all_projects):
                # Unselect all
                selected_projects.clear()
                for data in project_rows.values():
                    data["check_var"].set(False)
            else:
                # Select all
                selected_projects.clear()
                selected_projects.update(all_projects)
                for name, data in project_rows.items():
                    data["check_var"].set(True)
            _update_delete_bar()

        def _delete_single(project_name):
            confirm = messagebox.askyesno(
                "Hapus Project",
                f"Hapus project '{project_name}'?\n\nSemua data akan hilang permanen.",
                parent=popup,
            )
            if not confirm:
                return
            try:
                self.project_manager.delete_project(project_name)
                _build_project_rows()
                _update_delete_bar()
                remaining = self.project_manager.list_projects()
                ctk.CTkLabel  # refresh count label handled by rebuild
                if not remaining:
                    popup.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menghapus: {e}", parent=popup)

        def _delete_selected():
            if not selected_projects:
                return
            names = "\n".join(f"  • {p}" for p in sorted(selected_projects))
            confirm = messagebox.askyesno(
                "Hapus Project",
                f"Hapus {len(selected_projects)} project berikut?\n\n{names}\n\nSemua data akan hilang permanen.",
                parent=popup,
            )
            if not confirm:
                return
            errors = []
            for project_name in list(selected_projects):
                try:
                    self.project_manager.delete_project(project_name)
                except Exception as e:
                    errors.append(f"{project_name}: {e}")
            selected_projects.clear()
            _build_project_rows()
            _update_delete_bar()
            if errors:
                messagebox.showerror(
                    "Sebagian Gagal",
                    "Gagal menghapus:\n" + "\n".join(errors),
                    parent=popup,
                )
            remaining = self.project_manager.list_projects()
            if not remaining:
                popup.destroy()

        # ── Delete bar (muncul di select mode) ───────────────────────────────
        delete_bar = ctk.CTkFrame(popup, fg_color="#1e1e1e", corner_radius=8, height=48)
        delete_bar.pack_propagate(False)
        # tidak di-pack dulu, muncul saat select mode aktif

        select_all_btn = ctk.CTkButton(
            delete_bar,
            text="☑ Pilih Semua",
            width=120,
            height=34,
            fg_color="#3d4045",
            hover_color="#4a4d52",
            font=ctk.CTkFont(family="Arial", size=13),
            command=_toggle_select_all,
        )
        select_all_btn.pack(side="left", padx=(12, 8), pady=7)

        delete_selected_btn = ctk.CTkButton(
            delete_bar,
            text="🗑  Hapus yang Dipilih",
            width=180,
            height=34,
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
            state="disabled",
            command=_delete_selected,
        )
        delete_selected_btn.pack(side="right", padx=(8, 12), pady=7)

        # ── Footer ────────────────────────────────────────────────────────────
        sep2 = ctk.CTkFrame(popup, height=1, fg_color="#3d4045")
        sep2.pack(fill="x", padx=24, pady=(4, 0))

        ctk.CTkButton(
            popup,
            text="Batal",
            width=120,
            height=36,
            fg_color="#3d4045",
            hover_color="#4a4d52",
            font=ctk.CTkFont(family="Arial", size=14),
            command=popup.destroy,
        ).pack(pady=(10, 16))

        # ── Initial build ─────────────────────────────────────────────────────
        _build_project_rows()

    def _load_project(self, project_name, popup):

        popup.destroy()

        self.current_project = project_name

        state = self.project_manager.load_state(project_name)

        if not state:

            messagebox.showerror("Error", "Project state not found.")

            return

        # =====================================================
        # RESTORE BASIC STATE
        # =====================================================

        self.dataset_path = state.get("dataset_path")

        self.target_column = state.get("target_column")

        self.selected_models = state.get("selected_models", [])

        self.selected_features = state.get("selected_features", [])

        self.train_test_split = state.get("train_test_split", "80/20 (Recommended)")

        self.inferred_task = state.get("inferred_task")

        self.training_results = state.get("training_results")

        self.selected_best_model_name = state.get("selected_best_model_name")

        self.selected_best_model_path = state.get("selected_best_model_path")

        self.user_model_params = state.get("user_model_params", {})

        # =====================================================
        # OUTLIER STATE
        # =====================================================

        self.model_evaluation_details = state.get("model_evaluation_details", {})

        self.training_durations = state.get("training_durations", {})

        self.outlier_summary = state.get("outlier_summary", {})

        self.outlier_insights = state.get("outlier_insights", {})

        self.outlier_recommendations = state.get("outlier_recommendations", {})

        # Rebuild model_evaluation_details dari outlier_summary jika kosong
        # (backward compat untuk project lama)
        if not self.model_evaluation_details and self.outlier_summary:
            for model_name, summary in self.outlier_summary.items():
                self.model_evaluation_details[model_name] = {"summary": summary}

        # =====================================================
        # SHOW WORKSPACE DULU, BARU LOAD BERAT DI BACKGROUND
        # =====================================================

        self.show_workspace()

        if self.workspace_screen:

            self.workspace_screen.show_loading_overlay("Memuat project...")

            self.update()

        threading.Thread(
            target=self._load_project_heavy,
            args=(state,),
            daemon=True,
        ).start()

    def _load_project_heavy(self, state):
        """Load dataset & semua model di background thread, lalu hide overlay."""

        try:

            # =====================================================
            # LOAD SCORED DATASETS
            # =====================================================

            results_dir = os.path.join("projects", self.current_project, "results")

            if os.path.exists(results_dir):

                for file in os.listdir(results_dir):

                    if file.endswith("_scored.csv"):

                        model_name = (
                            file.replace("_scored.csv", "").replace("_", " ").title()
                        )

                        try:

                            path = os.path.join(results_dir, file)

                            self.scored_datasets[model_name] = pd.read_csv(path)

                            # Populate evaluation details dari scored dataset
                            if (
                                model_name not in self.model_evaluation_details
                                or not self.model_evaluation_details.get(model_name, {})
                            ):
                                self.model_evaluation_details[model_name] = {
                                    "scored_dataset": self.scored_datasets[model_name],
                                    "summary": self.outlier_summary.get(model_name, {}),
                                }
                            else:
                                # Inject scored_dataset ke evaluation details yang sudah ada
                                self.model_evaluation_details[model_name][
                                    "scored_dataset"
                                ] = self.scored_datasets[model_name]

                        except Exception as e:

                            print(f"Load scored dataset error ({model_name}): {e}")

            # =====================================================
            # LOAD DATASET
            # =====================================================

            if self.dataset_path and os.path.exists(self.dataset_path):

                self._update_loading_msg("Membaca dataset...")

                self.df = pd.read_csv(self.dataset_path)

            # =====================================================
            # LOAD SELECTED BEST MODEL (dan semua model dari artifacts)
            # =====================================================

            artifacts_dir = os.path.join("projects", self.current_project, "artifacts")

            # --- Load selected best model ---
            if self.selected_best_model_path and os.path.exists(
                self.selected_best_model_path
            ):

                self._update_loading_msg("Memuat model terpilih...")

                try:

                    loaded = joblib.load(self.selected_best_model_path)

                    self.selected_best_model = loaded["pipeline"]

                    self.label_encoder = loaded.get("label_encoder")

                    self.datetime_cols_for_prediction = loaded.get("datetime_cols", [])

                    self.feature_column_types = loaded.get("feature_column_types", {})

                    # Daftarkan ke all_trained_models
                    if self.selected_best_model_name:

                        self.all_trained_models[self.selected_best_model_name] = (
                            self.selected_best_model
                        )

                except Exception as e:

                    print(f"Load selected model error: {e}")

            # --- Load semua model lain dari artifacts folder ---
            if os.path.exists(artifacts_dir):

                for file in os.listdir(artifacts_dir):

                    if file.endswith("_selected_model.joblib"):

                        # Derive model name dari filename
                        model_name = file.replace("_selected_model.joblib", "")

                        if model_name in self.all_trained_models:
                            continue  # sudah di-load

                        self._update_loading_msg(f"Memuat model: {model_name}...")

                        try:

                            path = os.path.join(artifacts_dir, file)

                            loaded = joblib.load(path)

                            pipeline = loaded.get("pipeline")

                            if pipeline is not None:

                                self.all_trained_models[model_name] = pipeline

                                # Jika label_encoder belum ada, ambil dari sini
                                if self.label_encoder is None:
                                    self.label_encoder = loaded.get("label_encoder")

                                if not self.datetime_cols_for_prediction:
                                    self.datetime_cols_for_prediction = loaded.get(
                                        "datetime_cols", []
                                    )

                                if not self.feature_column_types:
                                    self.feature_column_types = loaded.get(
                                        "feature_column_types", {}
                                    )

                        except Exception as e:

                            print(f"Load artifact model error ({model_name}): {e}")

            # --- Pastikan semua model di training_results ada di all_trained_models
            #     Coba berbagai variasi nama file untuk kompatibilitas
            if self.training_results and os.path.exists(artifacts_dir):
                for model_name in self.training_results.keys():
                    if model_name in self.all_trained_models:
                        continue

                    # Daftar kandidat nama file yang mungkin (spasi asli, underscore, dll)
                    candidates = [
                        os.path.join(
                            artifacts_dir, f"{model_name}_selected_model.joblib"
                        ),
                        os.path.join(
                            artifacts_dir,
                            f"{model_name.replace(' ', '_')}_selected_model.joblib",
                        ),
                        os.path.join(
                            artifacts_dir,
                            f"{model_name.lower().replace(' ', '_')}_selected_model.joblib",
                        ),
                    ]
                    for candidate in candidates:
                        if os.path.exists(candidate):
                            try:
                                loaded = joblib.load(candidate)
                                p = loaded.get("pipeline")
                                if p:
                                    self.all_trained_models[model_name] = p
                                    self._update_loading_msg(
                                        f"Memuat model: {model_name}..."
                                    )
                                    if self.label_encoder is None:
                                        self.label_encoder = loaded.get("label_encoder")
                                    if not self.datetime_cols_for_prediction:
                                        self.datetime_cols_for_prediction = loaded.get(
                                            "datetime_cols", []
                                        )
                                    if not self.feature_column_types:
                                        self.feature_column_types = loaded.get(
                                            "feature_column_types", {}
                                        )
                                    break
                            except Exception as e:
                                print(
                                    f"Load fallback model error ({model_name}, {candidate}): {e}"
                                )

        except Exception as e:

            print(f"Load project heavy error: {e}")

        finally:

            self.after(0, self._finish_load_project)

    def _update_loading_msg(self, msg: str):
        """Thread-safe update loading message."""
        self.after(
            0,
            lambda: (
                self.workspace_screen
                and hasattr(self.workspace_screen, "loading_message_label")
                and self.workspace_screen.loading_message_label
                and self.workspace_screen.loading_message_label.winfo_exists()
                and self.workspace_screen.loading_message_label.configure(text=msg)
            ),
        )

    def _finish_load_project(self):
        """Dipanggil di main thread setelah load selesai."""

        if self.workspace_screen:

            self.workspace_screen.hide_loading_overlay()

        if not self.workspace_screen:
            return

        ws = self.workspace_screen

        # Aktifkan sidebar sesuai state
        if self.training_results:

            ws.sidebar_steps["results"].configure(state="normal")
            ws.sidebar_steps["try_model"].configure(state="normal")
            ws.sidebar_steps["export_model"].configure(state="normal")

            ws._go_to_main_step(ws.MAIN_STEP_RESULTS)

        elif self.df is not None and self.inferred_task:

            ws._go_to_main_step(ws.MAIN_STEP_CONFIGURE)

        elif self.df is not None:

            ws._go_to_main_step(ws.MAIN_STEP_UPLOAD)

        else:

            ws._go_to_main_step(ws.MAIN_STEP_UPLOAD)

    # =========================================================
    # DATASET
    # =========================================================

    def upload_dataset(self, filepath):

        if not self.current_project:

            messagebox.showerror("Error", "Create/Open project first.")

            return

        try:

            project_dir = os.path.join(
                "projects",
                self.current_project,
                "datasets",
            )

            os.makedirs(project_dir, exist_ok=True)

            filename = os.path.basename(filepath)

            destination = os.path.join(project_dir, filename)

            shutil.copy2(filepath, destination)

            self.dataset_path = destination

            self.df = pd.read_csv(destination)

            self._save_current_state("upload")

        except Exception as e:

            messagebox.showerror("Upload Error", str(e))

    # =========================================================
    # TRAINING
    # =========================================================

    def run_training(self):

        if self.is_training:

            messagebox.showwarning("Training", "Training already running.")

            return

        threading.Thread(
            target=self._run_training_thread,
            daemon=True,
        ).start()

    def _run_training_thread(self):

        try:

            self.is_training = True

            self.cancel_training = False

            if self.workspace_screen:
                self.after(0, lambda: self.workspace_screen.set_training_state(True))

            split_ratio = float(self.train_test_split.split("/")[0]) / 100.0

            trainer = MLTrainer(
                self.df,
                self.target_column,
                self.inferred_task,
                self.selected_features,
                user_model_params=self.user_model_params,
                test_size=1 - split_ratio,
            )

            if self.cancel_training:
                return

            (
                self.training_results,
                self.all_trained_models,
            ) = trainer.train(self.selected_models)

            if self.cancel_training:
                self.training_results = None
                self.all_trained_models = {}
                return

            self.score_stats = getattr(trainer, "score_stats", {})
            self.feature_stats = getattr(trainer, "feature_stats", {})

            # =================================================
            # SAVE EVALUATION DATA
            # =================================================

            self.model_evaluation_details = trainer.model_evaluation_details

            self.scored_datasets = trainer.scored_datasets

            self.training_durations = trainer.training_durations

            self._save_scored_datasets()

            # =================================================
            # SUMMARY / INSIGHT / RECOMMENDATION
            # =================================================

            self.outlier_summary = {}

            self.outlier_insights = {}

            self.outlier_recommendations = {}

            for model_name, detail in self.model_evaluation_details.items():

                summary = detail.get("summary", {})

                self.outlier_summary[model_name] = summary

                self.outlier_insights[model_name] = summary.get("insights", [])

                self.outlier_recommendations[model_name] = summary.get(
                    "recommendations", []
                )

            # =================================================
            # LABEL ENCODER
            # =================================================

            self.label_encoder = trainer.label_encoder

            self.datetime_cols_for_prediction = trainer.datetime_cols

            # =================================================
            # FEATURE TYPES
            # =================================================

            valid_features = [
                f for f in self.selected_features if f in trainer.df.columns
            ]

            if valid_features:

                self.feature_column_types = get_column_types(
                    trainer.df[valid_features],
                    trainer.datetime_cols,
                )

            # =================================================
            # AUTO SELECT BEST MODEL  &  SAVE ALL MODELS
            # =================================================

            for model_name, pipeline in self.all_trained_models.items():
                self._save_model_artifact(model_name, pipeline)

            if (
                trainer.best_model_name
                and trainer.best_model_name in self.all_trained_models
            ):

                self.set_selected_best_model(
                    trainer.best_model_name,
                    self.all_trained_models[trainer.best_model_name],
                )

            # =================================================
            # SAVE STATE
            # =================================================

            self._save_current_state("results")

            # =================================================
            # SHOW RESULTS
            # =================================================

            self.after(0, self.show_results_screen)

        except Exception as e:

            if not self.cancel_training:

                error_message = str(e)

                def _show_error_and_go_back(msg):
                    messagebox.showerror("Training Error", msg)
                    if self.workspace_screen:
                        self.workspace_screen._go_to_main_step(
                            self.workspace_screen.MAIN_STEP_CONFIGURE
                        )

                self.after(
                    0,
                    lambda msg=error_message: _show_error_and_go_back(msg),
                )

        finally:

            self.is_training = False
            self.cancel_training = False

            if self.workspace_screen:

                self.after(
                    0,
                    lambda: self.workspace_screen.set_training_state(False),
                )

    # =========================================================
    # MODEL
    # =========================================================

    def _save_model_artifact(self, model_name: str, pipeline):
        """Simpan satu model ke artifacts folder."""

        artifacts_dir = os.path.join("projects", self.current_project, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        model_path = os.path.join(
            artifacts_dir,
            f"{model_name}_selected_model.joblib",
        )

        try:
            from core.trainer import _sanitize_pipeline

            clean_pipeline = _sanitize_pipeline(pipeline)

            # ── label_encoder ─────────────────────────────────────────────────
            safe_label_encoder = None
            if self.label_encoder is not None:
                try:
                    safe_label_encoder = {
                        "classes": [str(c) for c in self.label_encoder.classes_]
                    }
                except Exception:
                    safe_label_encoder = None

            # ── evaluation_details: buang scored_dataset ──────────────────────
            raw_detail = self.model_evaluation_details.get(model_name, {})
            safe_detail = {k: v for k, v in raw_detail.items() if k != "scored_dataset"}

            # ── feature_statistics: ekstrak dari StandardScaler ───────────────
            feature_statistics = _extract_feature_statistics(clean_pipeline)

            joblib.dump(
                {
                    "pipeline": clean_pipeline,
                    "label_encoder": safe_label_encoder,
                    "datetime_cols": list(self.datetime_cols_for_prediction),
                    "feature_column_types": self.feature_column_types,
                    "task_type": self.inferred_task,
                    "evaluation_details": safe_detail,
                    "feature_statistics": feature_statistics,
                },
                model_path,
            )

            return model_path

        except Exception as e:
            print(f"Save model artifact error ({model_name}): {e}")
            return None

    def set_selected_best_model(self, model_name, model_pipeline):

        self.selected_best_model_name = model_name

        self.selected_best_model = model_pipeline

        model_path = self._save_model_artifact(model_name, model_pipeline)

        if model_path:

            self.selected_best_model_path = model_path

        else:

            messagebox.showerror(
                "Save Model Error", f"Gagal menyimpan model '{model_name}'."
            )

    # =========================================================
    # RESULTS
    # =========================================================

    def show_results_screen(self):

        if self.workspace_screen:

            self.workspace_screen._go_to_main_step(
                self.workspace_screen.MAIN_STEP_RESULTS
            )

    # =========================================================
    # EXPORT
    # =========================================================

    def export_scored_dataset(self, model_name):

        try:

            if model_name not in self.scored_datasets:

                messagebox.showerror("Error", "Scored dataset not found.")

                return

            df = self.scored_datasets[model_name]

            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv")],
            )

            if not path:
                return

            df.to_csv(path, index=False)

            messagebox.showinfo("Success", "Scored dataset exported.")

        except Exception as e:

            messagebox.showerror("Export Error", str(e))

    def export_top_outliers(self, model_name, top_n=20):

        try:

            if model_name not in self.scored_datasets:

                messagebox.showerror("Error", "Scored dataset not found.")

                return

            df = self.scored_datasets[model_name]

            top_df = df.nsmallest(top_n, "outlier_score")

            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv")],
            )

            if not path:
                return

            top_df.to_csv(path, index=False)

            messagebox.showinfo("Success", "Top outliers exported.")

        except Exception as e:

            messagebox.showerror("Export Error", str(e))

    # =========================================================
    # STATE
    # =========================================================

    def _get_current_state_dict(self):

        return {
            "dataset_path": self.dataset_path,
            "target_column": self.target_column,
            "selected_models": self.selected_models,
            "selected_features": self.selected_features,
            "train_test_split": self.train_test_split,
            "inferred_task": self.inferred_task,
            "training_results": self.training_results,
            "selected_best_model_name": self.selected_best_model_name,
            "selected_best_model_path": self.selected_best_model_path,
            "training_durations": self.training_durations,
            # model_evaluation_details: hanya simpan bagian yang JSON-serializable
            # (scored_dataset adalah DataFrame, tidak bisa di-JSON)
            "model_evaluation_details": {
                k: {
                    ik: iv
                    for ik, iv in v.items()
                    if ik != "scored_dataset"  # skip DataFrame
                }
                for k, v in self.model_evaluation_details.items()
            },
            "outlier_summary": self.outlier_summary,
            "outlier_insights": self.outlier_insights,
            "outlier_recommendations": self.outlier_recommendations,
            "user_model_params": self.user_model_params,
        }

    def _save_current_state(self, last_phase):

        if not self.current_project:
            return

        state = self._get_current_state_dict()

        state["last_phase"] = last_phase

        self.project_manager.save_state(
            self.current_project,
            state,
        )

    # =====================================================
    # SAVE SCORED DATASETS
    # =====================================================

    def _save_scored_datasets(self):

        try:

            if not self.current_project:
                return

            result_dir = os.path.join("projects", self.current_project, "results")

            os.makedirs(result_dir, exist_ok=True)

            for model_name, df in self.scored_datasets.items():

                safe_name = model_name.replace(" ", "_").lower()

                path = os.path.join(result_dir, f"{safe_name}_scored.csv")

                df.to_csv(path, index=False)

        except Exception as e:

            print(f"Save scored dataset error: {e}")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    app = GalleMLStudio()

    app.mainloop()
