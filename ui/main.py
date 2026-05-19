"""
ui/main.py
Entry point UI layer.

Ekspor:
  - StartScreen   → halaman awal (New / Open Project)
  - WorkspaceScreen → shell workspace yang mengkomposisi semua page
"""

import customtkinter as ctk
import pandas as pd
from tkinter import messagebox

from ui.pages.start_page import StartScreen
from ui.pages.upload_page import UploadPage
from ui.pages.task_page import TaskPage
from ui.pages.configure_page import ConfigurePage
from ui.pages.results_page import ResultsPage
from ui.pages.try_model_page import TryModelPage
from ui.pages.export_page import ExportPage
from ui.pages.metrics_popup import MetricsPopupMixin

# Re-export StartScreen agar app.py cukup import dari ui.main
__all__ = ["StartScreen", "WorkspaceScreen"]


class WorkspaceScreen:
    """
    Shell WorkspaceScreen:
    - Menyiapkan layout (sidebar, main_area, nav_area, overlay)
    - Mendelegasikan setiap step ke page yang sesuai
    - Menyimpan state navigasi & training overlay
    """

    # Step indices
    MAIN_STEP_UPLOAD = 0
    MAIN_STEP_TASK_SELECTION = 1
    MAIN_STEP_CONFIGURE = 2
    MAIN_STEP_TRAIN = 3
    MAIN_STEP_RESULTS = 4
    MAIN_STEP_TRY_MODEL = 5
    MAIN_STEP_EXPORT_MODEL = 6

    def __init__(self, app):
        self.app = app

        # Shared UI references (diisi di render())
        self.content_frame = None
        self.main_area = None
        self.nav_area = None
        self.sidebar_steps = {}

        # Overlay references
        self.training_overlay = None
        self.training_spinner_label = None
        self.training_time_label = None
        self.training_animation_job = None
        self.training_start_time = None

        self.loading_overlay = None
        self.loading_message_label = None
        self.loading_spinner_label = None
        self.loading_animation_job = None

        # Configure wizard state (dipakai configure_page via ws reference)
        self.target_var = None
        self.split_var = None
        self.feature_vars = {}
        self.model_vars = {}
        self.model_param_widgets = {}
        self.model_param_frames = {}
        self._step_nav_buttons = []
        self._current_sub_step = 0
        self._placeholder = "-- Select Target --"
        self._target_selected = False
        self._target_next_btn = None
        self.target_warning_label = None
        self.target_preview = None
        self.train_btn = None
        self._wizard_area = None
        self._summary_box = None

        self._current_main_step = 0

        # Instantiate pages
        self._upload_page = UploadPage(self)
        self._task_page = TaskPage(self)
        self._configure_page = ConfigurePage(self)
        self._results_page = ResultsPage(self)
        self._try_model_page = TryModelPage(self)
        self._export_page = ExportPage(self)
        self._metrics_popup = MetricsPopupMixin(self)

    # =========================================================================
    #  RENDER
    # =========================================================================
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

    # =========================================================================
    #  SIDEBAR
    # =========================================================================
    def _setup_sidebar(self, sidebar):

        sidebar.grid_columnconfigure(0, weight=1)
        for r in range(12):
            sidebar.grid_rowconfigure(r, weight=0)
        sidebar.grid_rowconfigure(9, weight=1)

        import os
        from PIL import Image

        logo_path = os.path.join("assets", "logo_with_text_dark_mode.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join("assets", "logo.png")

        if os.path.exists(logo_path):
            logo_img = ctk.CTkImage(
                light_image=Image.open(logo_path),
                dark_image=Image.open(logo_path),
                size=(160, 58),
            )
            ctk.CTkLabel(
                sidebar,
                image=logo_img,
                text="",
            ).grid(row=0, column=0, pady=(24, 32))
        else:
            # Fallback teks
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

        # Project info
        info_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        info_frame.grid(row=8, column=0, pady=(20, 8), padx=20, sticky="ew")
        info_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            info_frame,
            text=f"Project: {self.app.current_project or 'None'}",
            font=ctk.CTkFont(family="Arial", size=14),
        ).grid(row=0, column=0)

        # Tombol Hapus Project
        ctk.CTkButton(
            sidebar,
            text="🗑️  Hapus Project",
            width=220,
            height=38,
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
            command=self._confirm_delete_project,
        ).grid(row=9, column=0, pady=(0, 16), padx=20, sticky="s")

    # ── Hapus Project ─────────────────────────────────────────────────────────
    def _confirm_delete_project(self):
        project = self.app.current_project
        if not project:
            return

        popup = ctk.CTkToplevel(self.app)
        popup.title("Hapus Project")
        popup.resizable(False, False)
        popup.grab_set()
        popup.transient(self.app)
        popup.lift()

        w, h = 440, 240
        self.app.update_idletasks()
        x = self.app.winfo_rootx() + self.app.winfo_width() // 2 - w // 2
        y = self.app.winfo_rooty() + self.app.winfo_height() // 2 - h // 2
        popup.geometry(f"{w}x{h}+{x}+{y}")

        ctk.CTkLabel(
            popup,
            text="🗑️  Hapus Project?",
            font=ctk.CTkFont(family="Arial", size=20, weight="bold"),
        ).pack(pady=(30, 8))

        ctk.CTkLabel(
            popup,
            text=f'Project  "{project}"  akan dihapus permanen.\n'
            "Semua dataset, model, dan hasil training akan hilang.",
            font=ctk.CTkFont(family="Arial", size=13),
            text_color="gray60",
            justify="center",
        ).pack(pady=(0, 24))

        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack()

        ctk.CTkButton(
            btn_frame,
            text="Ya, Hapus",
            width=150,
            height=42,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            font=ctk.CTkFont(weight="bold"),
            command=lambda: self._do_delete_project(popup),
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            btn_frame,
            text="Batal",
            width=150,
            height=42,
            fg_color="#3d4045",
            hover_color="#4a4d52",
            command=popup.destroy,
        ).pack(side="left", padx=10)

    def _do_delete_project(self, popup):
        popup.destroy()
        project = self.app.current_project
        try:
            self.app.project_manager.delete_project(project)
            messagebox.showinfo(
                "Berhasil",
                f"Project '{project}' berhasil dihapus.",
            )
            # Reset state dan kembali ke start screen
            self.app.current_project = None
            self.app.df = None
            self.app.training_results = None
            self.app.all_trained_models = {}
            self.app.selected_best_model = None
            self.app.selected_best_model_name = None
            self.app.selected_best_model_path = None
            self.app.selected_features = []
            self.app.selected_models = []
            self.app.inferred_task = None
            self.app.dataset_path = None
            self.app.show_start_screen()
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menghapus project: {e}")

    # =========================================================================
    #  LOADING OVERLAY
    # =========================================================================
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

    # =========================================================================
    #  TRAINING OVERLAY
    # =========================================================================
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

    def set_training_state(self, is_training: bool):

        if is_training:
            self.training_overlay.grid(row=1, column=1, sticky="nsew")
            self.training_overlay.tkraise()
            self.training_start_time = pd.Timestamp.now()
            self._update_training_animation()

            # Disable tombol via referensi langsung ke configure_page
            self._set_train_btn_state("disabled")

        else:
            # Cancel animasi
            if self.training_animation_job:
                try:
                    self.training_overlay.after_cancel(self.training_animation_job)
                except Exception:
                    pass
                self.training_animation_job = None

            self.training_overlay.grid_forget()
            self.training_overlay.lower()

            self._set_train_btn_state("normal")

    def _set_train_btn_state(self, state: str):
        try:
            cp = getattr(self, "_configure_page", None)

            if cp and hasattr(cp, "train_btn"):
                btn = cp.train_btn
                if btn and btn.winfo_exists():
                    btn.configure(state=state)
                    return

        except Exception as e:
            print(f"[_set_train_btn_state] {e}")

        # Fallback
        try:
            if self.train_btn and self.train_btn.winfo_exists():
                self.train_btn.configure(state=state)
        except Exception:
            pass

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

    def _confirm_cancel_training(self):
        popup = ctk.CTkToplevel(self.app)
        popup.title("Cancel Training")
        popup.resizable(False, False)
        popup.grab_set()
        popup.transient(self.app)

        w, h = 400, 200
        self.app.update_idletasks()
        x = self.app.winfo_rootx() + self.app.winfo_width() // 2 - w // 2
        y = self.app.winfo_rooty() + self.app.winfo_height() // 2 - h // 2
        popup.geometry(f"{w}x{h}+{x}+{y}")

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

        self._set_cancelling_overlay()
        self._wait_training_done_then_go_configure()

    def _set_cancelling_overlay(self):
        """Ubah overlay training jadi tampilan 'Membatalkan...'"""
        # Stop animasi spinner
        if self.training_animation_job:
            try:
                self.training_overlay.after_cancel(self.training_animation_job)
            except Exception:
                pass
            self.training_animation_job = None

        # Update teks spinner dan label
        if self.training_spinner_label and self.training_spinner_label.winfo_exists():
            self.training_spinner_label.configure(text="⏹️")

        # Cari label "Training Models..." dan update
        # Overlay tetap tampil, hanya konten yang berubah
        container = None
        for widget in self.training_overlay.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                container = widget
                break

        if container:
            for widget in container.winfo_children():
                try:
                    text = widget.cget("text")
                    if "Training Models" in str(text):
                        widget.configure(text="Membatalkan training...")
                    elif "Elapsed" in str(text):
                        widget.configure(
                            text="Menunggu proses selesai...",
                            text_color="#f39c12",
                        )
                    elif "Cancel" in str(text):
                        widget.configure(state="disabled", text="Sedang membatalkan...")
                except Exception:
                    pass

        # Mulai animasi dots
        self._cancelling_dots = 0
        self._update_cancelling_animation()

    def _update_cancelling_animation(self):
        """Animasi titik-titik saat menunggu cancel selesai."""
        if not self.app.cancel_training:
            return

        if not self.training_spinner_label.winfo_exists():
            return

        spinners = ["⏹️ ", " ⏹️", "⏹️ "]
        self.training_spinner_label.configure(
            text=spinners[self._cancelling_dots % len(spinners)]
        )
        self._cancelling_dots += 1
        self._cancelling_animation_job = self.app.after(
            300, self._update_cancelling_animation
        )

    def _wait_training_done_then_go_configure(self, attempts=0):
        """Poll sampai is_training = False, baru pindah ke Configure."""
        if not self.app.is_training:
            # Stop animasi cancel
            if hasattr(self, "_cancelling_animation_job"):
                try:
                    self.app.after_cancel(self._cancelling_animation_job)
                except Exception:
                    pass

            self.app.cancel_training = False
            self.set_training_state(False)  # sembunyikan overlay
            self._go_to_main_step(self.MAIN_STEP_CONFIGURE)
            return

        if attempts >= 100:
            if hasattr(self, "_cancelling_animation_job"):
                try:
                    self.app.after_cancel(self._cancelling_animation_job)
                except Exception:
                    pass
            self.app.is_training = False
            self.app.cancel_training = False
            self.set_training_state(False)
            self._go_to_main_step(self.MAIN_STEP_CONFIGURE)
            return

        self.app.after(
            100, lambda: self._wait_training_done_then_go_configure(attempts + 1)
        )

    # =========================================================================
    #  NAVIGATION
    # =========================================================================
    def _go_to_main_step(self, step_idx):
        self._current_main_step = step_idx
        self.clear_content()
        self._update_sidebar_highlight()

        if step_idx == self.MAIN_STEP_UPLOAD:
            self._upload_page.show_welcome()
            phase = "upload"

        elif step_idx == self.MAIN_STEP_TASK_SELECTION:
            if not hasattr(self.app, "df") or self.app.df is None:
                messagebox.showwarning("Warning", "Please upload a dataset first.")
                self._go_to_main_step(self.MAIN_STEP_UPLOAD)
                return
            self._task_page.show_task_selection_screen()
            phase = "task_selection"

        elif step_idx == self.MAIN_STEP_CONFIGURE:
            if not hasattr(self.app, "df") or self.app.df is None:
                messagebox.showwarning("Warning", "Please upload a dataset first.")
                self._go_to_main_step(self.MAIN_STEP_UPLOAD)
                return
            if not hasattr(self.app, "inferred_task") or not self.app.inferred_task:
                messagebox.showwarning("Warning", "Please select a task type first.")
                self._go_to_main_step(self.MAIN_STEP_TASK_SELECTION)
                return
            self._configure_page.show_target_screen()
            phase = "configure"

        elif step_idx == self.MAIN_STEP_TRAIN:
            phase = "train"

        elif step_idx == self.MAIN_STEP_RESULTS:
            self._results_page.show_results_screen()
            phase = "results"

        elif step_idx == self.MAIN_STEP_TRY_MODEL:
            self._try_model_page.show_try_model_screen()
            phase = "try_model"

        elif step_idx == self.MAIN_STEP_EXPORT_MODEL:
            self._export_page.show_export_model_screen()
            phase = "export_model"

        else:
            phase = "unknown"

        self.app._save_current_state(phase)

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

    # =========================================================================
    #  HELPERS
    # =========================================================================
    def clear_content(self):
        for widget in self.main_area.winfo_children():
            widget.destroy()
        for widget in self.nav_area.winfo_children():
            widget.destroy()

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

    # ── delegate show_metrics_explanation_popup ke MetricsPopupMixin ─────────
    def show_metrics_explanation_popup(self):
        self._metrics_popup.show_metrics_explanation_popup()
