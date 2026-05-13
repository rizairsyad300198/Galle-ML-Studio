# ================================
# app.py
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

from ui.screens import (
    StartScreen,
    WorkspaceScreen,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class GalleMLStudio(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title("Galle ML Studio")
        self.geometry("1600x950")

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
        # NEW OUTLIER EVALUATION STATE
        # =====================================================

        self.model_evaluation_details = {}

        self.scored_datasets = {}

        self.training_durations = {}

        self.outlier_summary = {}

        self.outlier_insights = {}

        self.outlier_recommendations = {}

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

        popup.geometry("400x500")

        popup.grab_set()

        ctk.CTkLabel(
            popup,
            text="Select Project",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(pady=30)

        for project in projects:

            ctk.CTkButton(
                popup,
                text=project,
                width=300,
                height=45,
                command=lambda p=project: self._load_project(p, popup),
            ).pack(pady=8)

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

        self.train_test_split = state.get("train_test_split", "80/20")

        self.inferred_task = state.get("inferred_task")

        self.training_results = state.get("training_results")

        self.selected_best_model_name = state.get("selected_best_model_name")

        self.selected_best_model_path = state.get("selected_best_model_path")

        # =====================================================
        # NEW OUTLIER STATE
        # =====================================================

        self.model_evaluation_details = state.get("model_evaluation_details", {})

        self.training_durations = state.get("training_durations", {})

        self.outlier_summary = state.get("outlier_summary", {})

        self.outlier_insights = state.get("outlier_insights", {})

        self.outlier_recommendations = state.get("outlier_recommendations", {})

        # =====================================================
        # SHOW WORKSPACE DULU, BARU LOAD BERAT DI BACKGROUND
        # =====================================================

        self.show_workspace()

        # Tampilkan loading overlay setelah workspace siap
        if self.workspace_screen:

            self.workspace_screen.show_loading_overlay("Memuat project...")

            self.update()

        # Jalankan load berat di thread terpisah
        threading.Thread(
            target=self._load_project_heavy,
            args=(state,),
            daemon=True,
        ).start()

    def _load_project_heavy(self, state):
        """Load dataset & model di background thread, lalu hide overlay."""

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

                        except Exception as e:

                            print(f"Load scored dataset error: {e}")

            # =====================================================
            # LOAD DATASET
            # =====================================================

            if self.dataset_path and os.path.exists(self.dataset_path):

                self.after(
                    0,
                    lambda: self.workspace_screen
                    and self.workspace_screen.loading_message_label
                    and self.workspace_screen.loading_message_label.configure(
                        text="Membaca dataset..."
                    ),
                )

                self.df = pd.read_csv(self.dataset_path)

            # =====================================================
            # LOAD MODEL
            # =====================================================

            if self.selected_best_model_path and os.path.exists(
                self.selected_best_model_path
            ):

                self.after(
                    0,
                    lambda: self.workspace_screen
                    and self.workspace_screen.loading_message_label
                    and self.workspace_screen.loading_message_label.configure(
                        text="Memuat model..."
                    ),
                )

                loaded = joblib.load(self.selected_best_model_path)

                self.selected_best_model = loaded["pipeline"]

                self.label_encoder = loaded.get("label_encoder")

                self.datetime_cols_for_prediction = loaded.get("datetime_cols", [])

                self.feature_column_types = loaded.get("feature_column_types", {})

        except Exception as e:

            print(f"Load project error: {e}")

        finally:

            # Setelah semua selesai, hide overlay dan navigate ke fase yang benar
            self.after(0, self._finish_load_project)

    def _finish_load_project(self):
        """Dipanggil di main thread setelah load selesai."""

        if self.workspace_screen:

            self.workspace_screen.hide_loading_overlay()

        last_phase = self.project_manager.load_state(self.current_project).get(
            "last_phase", "upload"
        )

        if not self.workspace_screen:
            return

        ws = self.workspace_screen

        # Aktifkan sidebar sesuai state
        if self.training_results:

            ws.sidebar_steps["results"].configure(state="normal")
            ws.sidebar_steps["try_model"].configure(state="normal")
            ws.sidebar_steps["export_model"].configure(state="normal")

            # Selalu navigate ke results jika training sudah ada
            ws._go_to_main_step(ws.MAIN_STEP_RESULTS)

        elif self.df is not None and self.inferred_task:

            # Training belum selesai/dibatalkan, kembali ke configure
            ws._go_to_main_step(ws.MAIN_STEP_CONFIGURE)

        elif self.df is not None:

            ws._go_to_main_step(ws.MAIN_STEP_UPLOAD)  # show dataset preview

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

            # Cek cancel sebelum mulai
            if self.cancel_training:
                return

            (
                self.training_results,
                self.all_trained_models,
            ) = trainer.train(self.selected_models)

            # Cek cancel setelah training
            if self.cancel_training:
                self.training_results = None
                self.all_trained_models = {}
                return

            self.score_stats = trainer.score_stats

            self.feature_stats = trainer.feature_stats

            # =================================================
            # SAVE NEW EVALUATION DATA
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
            # AUTO SELECT BEST MODEL
            # =================================================

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

    def set_selected_best_model(self, model_name, model_pipeline):

        self.selected_best_model_name = model_name

        self.selected_best_model = model_pipeline

        artifacts_dir = os.path.join(
            "projects",
            self.current_project,
            "artifacts",
        )

        os.makedirs(artifacts_dir, exist_ok=True)

        model_path = os.path.join(
            artifacts_dir,
            f"{model_name}_selected_model.joblib",
        )

        try:

            joblib.dump(
                {
                    "pipeline": self.selected_best_model,
                    "label_encoder": self.label_encoder,
                    "datetime_cols": self.datetime_cols_for_prediction,
                    "feature_column_types": self.feature_column_types,
                    "evaluation_details": self.model_evaluation_details.get(model_name),
                },
                model_path,
            )

            self.selected_best_model_path = model_path

        except Exception as e:

            messagebox.showerror("Save Model Error", str(e))

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
            "outlier_summary": self.outlier_summary,
            "outlier_insights": self.outlier_insights,
            "outlier_recommendations": self.outlier_recommendations,
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
