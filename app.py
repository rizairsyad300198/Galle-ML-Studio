# ================================
# app.py  (FIXED)
# ================================

import customtkinter as ctk
import pandas as pd
import os
import shutil
import threading
import joblib

from tkinter import filedialog, messagebox

from core.project_manager import ProjectManager
from core.trainer import (
    MLTrainer,
    get_column_types,
)

from ui.main import StartScreen, WorkspaceScreen

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class GalleMLStudio(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title("Galle ML Studio")

        # Set geometry awal sebagai fallback
        self.geometry("1600x950")

        # Maximize window setelah event loop siap
        # (harus via after() agar bekerja di Windows — state("zoomed") di __init__ sering diabaikan)
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

        # Hitung tinggi dinamis berdasarkan jumlah project (min 300, max 600)
        popup_w = 460
        popup_h = min(600, max(300, 120 + len(projects) * 62))

        # Center terhadap parent window
        self.update_idletasks()
        px = self.winfo_rootx()
        py = self.winfo_rooty()
        pw = self.winfo_width()
        ph = self.winfo_height()
        x = px + (pw // 2) - (popup_w // 2)
        y = py + (ph // 2) - (popup_h // 2)
        popup.geometry(f"{popup_w}x{popup_h}+{x}+{y}")

        ctk.CTkLabel(
            popup,
            text="📂  Open Project",
            font=ctk.CTkFont(family="Arial", size=26, weight="bold"),
        ).pack(pady=(30, 20))

        scroll = ctk.CTkScrollableFrame(popup, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        for project in projects:
            ctk.CTkButton(
                scroll,
                text=project,
                width=380,
                height=48,
                font=ctk.CTkFont(family="Arial", size=15),
                fg_color="#2b2d30",
                hover_color="#3498db",
                anchor="w",
                command=lambda p=project: self._load_project(p, popup),
            ).pack(pady=5)

        ctk.CTkButton(
            popup,
            text="Batal",
            width=120,
            height=36,
            fg_color="#3d4045",
            hover_color="#4a4d52",
            font=ctk.CTkFont(family="Arial", size=14),
            command=popup.destroy,
        ).pack(pady=(0, 20))

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

                self.workspace_screen.set_training_state(True)

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

                self.after(
                    0,
                    lambda msg=error_message: messagebox.showerror(
                        "Training Error", msg
                    ),
                )

        finally:

            self.is_training = False

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

            joblib.dump(
                {
                    "pipeline": pipeline,
                    "label_encoder": self.label_encoder,
                    "datetime_cols": self.datetime_cols_for_prediction,
                    "feature_column_types": self.feature_column_types,
                    "evaluation_details": self.model_evaluation_details.get(model_name),
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
