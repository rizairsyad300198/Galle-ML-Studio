"""
ui/pages/results_page.py
Step 4 — Training Results viewer.
"""

import customtkinter as ctk
from tkinter import messagebox
import pandas as pd

from ui.widgets import BasePage


class ResultsPage(BasePage):
    def show_results_screen(self):
        self.clear_content()

        results = self.app.training_results

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
            command=self.ws.show_metrics_explanation_popup,
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
            model_in_memory = model_name in all_trained_models
            badge = "\u2705 " if model_in_memory else "\u26a0\ufe0f "
            btn = ctk.CTkButton(
                model_list_frame,
                text=badge + model_name,
                command=lambda m=model_name: self._on_model_button_click(m),
                font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
                fg_color="#2B2D30",
                hover_color="#3A3D40",
                anchor="w",
                height=48,
                corner_radius=8,
            )
            btn.pack(fill="x", pady=5, padx=5)
            self.model_buttons[model_name] = btn

        ctk.CTkLabel(
            model_list_frame,
            text="\u2705 siap dipakai  /  \u26a0\ufe0f perlu re-train",
            font=ctk.CTkFont(family="Arial", size=11),
            text_color="gray50",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(8, 0))

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
            text="\u2190 Back: Configure",
            width=180,
            height=44,
            fg_color="#3d4045",
            hover_color="#4a4d52",
            font=ctk.CTkFont(family="Arial", size=15),
            command=lambda: self._go_to_main_step(self.ws.MAIN_STEP_CONFIGURE),
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))

        mid_frame = ctk.CTkFrame(self.nav_area, fg_color="transparent")
        mid_frame.grid(row=0, column=1, sticky="ew")
        mid_frame.grid_columnconfigure(0, weight=1)

        self.selected_model_display_label = ctk.CTkLabel(
            mid_frame,
            text="\u2190 Klik model di kiri untuk memilih",
            font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
            text_color="#3498db",
        )
        self.selected_model_display_label.grid(row=0, column=0, pady=(0, 4))

        self.select_model_btn = ctk.CTkButton(
            mid_frame,
            text="\u2705 Gunakan Model Ini",
            width=240,
            height=38,
            font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
            fg_color="#27ae60",
            hover_color="#219a52",
            command=self._confirm_model_selection,
            state="disabled",
        )
        self.select_model_btn.grid(row=1, column=0)

        self.confirm_status_label = ctk.CTkLabel(
            mid_frame,
            text="",
            font=ctk.CTkFont(family="Arial", size=11),
            text_color="gray60",
        )
        self.confirm_status_label.grid(row=2, column=0, pady=(2, 0))

        can_proceed = bool(
            self.app.selected_best_model_name
            and getattr(self.app, "selected_best_model", None)
        )
        self.next_button_results = ctk.CTkButton(
            self.nav_area,
            text="Next: Try Model \u2192",
            width=200,
            height=44,
            fg_color="#3498db",
            hover_color="#2980b9",
            font=ctk.CTkFont(family="Arial", size=15, weight="bold"),
            command=lambda: self._go_to_main_step(self.ws.MAIN_STEP_TRY_MODEL),
            state="normal" if can_proceed else "disabled",
        )
        self.next_button_results.grid(row=0, column=2, sticky="e", padx=(8, 0))

        if (
            self.app.selected_best_model_name
            and self.app.selected_best_model_name in self.model_buttons
        ):
            self._on_model_button_click(self.app.selected_best_model_name)

        self._poll_model_loading()

    def _poll_model_loading(self):
        if not hasattr(self, "model_buttons") or not self.model_buttons:
            return
        try:
            first_btn = next(iter(self.model_buttons.values()))
            if not first_btn.winfo_exists():
                return
        except Exception:
            return

        all_trained = getattr(self.app, "all_trained_models", {})
        results = getattr(self.app, "training_results", {}) or {}

        for model_name, btn in self.model_buttons.items():
            if not btn.winfo_exists():
                return
            model_in_memory = model_name in all_trained
            badge = "\u2705 " if model_in_memory else "\u26a0\ufe0f "
            expected_text = badge + model_name
            if btn.cget("text") != expected_text:
                btn.configure(text=expected_text)

        if (
            self.selected_model_name_for_display
            and self.selected_model_name_for_display in all_trained
        ):
            if (
                hasattr(self, "select_model_btn")
                and self.select_model_btn.winfo_exists()
            ):
                if self.select_model_btn.cget("state") == "disabled":
                    self.select_model_btn.configure(state="normal")
                    if (
                        hasattr(self, "confirm_status_label")
                        and self.confirm_status_label.winfo_exists()
                    ):
                        self.confirm_status_label.configure(
                            text="Model siap \u2014 klik 'Gunakan Model Ini'",
                            text_color="#2ecc71",
                        )

        all_loaded = all(m in all_trained for m in results.keys())
        if not all_loaded:
            self.nav_area.after(500, self._poll_model_loading)

    def _on_model_button_click(self, model_name):
        if (
            self.selected_model_name_for_display
            and self.selected_model_name_for_display in self.model_buttons
        ):
            prev_btn = self.model_buttons[self.selected_model_name_for_display]
            if prev_btn.winfo_exists():
                prev_btn.configure(fg_color="#2B2D30", text_color="white")

        if model_name not in self.model_buttons:
            return

        all_trained = getattr(self.app, "all_trained_models", {})
        model_in_memory = model_name in all_trained

        self.model_buttons[model_name].configure(fg_color="#3498db", text_color="white")
        self.selected_model_name_for_display = model_name
        self.selected_model_display_label.configure(text=f"Model dipilih: {model_name}")

        if hasattr(self, "select_model_btn") and self.select_model_btn.winfo_exists():
            if model_in_memory:
                self.select_model_btn.configure(state="normal")
                if (
                    hasattr(self, "confirm_status_label")
                    and self.confirm_status_label.winfo_exists()
                ):
                    self.confirm_status_label.configure(
                        text="\u2705 Model ada di memori \u2014 siap digunakan",
                        text_color="#2ecc71",
                    )
            else:
                self.select_model_btn.configure(state="disabled")
                if (
                    hasattr(self, "confirm_status_label")
                    and self.confirm_status_label.winfo_exists()
                ):
                    self.confirm_status_label.configure(
                        text="\u26a0\ufe0f Model tidak ada di disk \u2014 perlu re-training",
                        text_color="#f39c12",
                    )

        metrics = self.app.training_results.get(model_name, {})
        evaluation_detail = self.app.model_evaluation_details.get(model_name, {})
        summary = evaluation_detail.get("summary", {})

        if not summary and hasattr(self.app, "outlier_summary"):
            summary = self.app.outlier_summary.get(model_name, {})
        if not summary and metrics:
            summary = metrics

        scored_df = evaluation_detail.get("scored_dataset", pd.DataFrame())
        if scored_df.empty and hasattr(self.app, "scored_datasets"):
            scored_df = self.app.scored_datasets.get(model_name, pd.DataFrame())

        # ── Ambil feature_statistics dari joblib jika tersedia ────────────────
        feature_statistics = self._load_feature_statistics(model_name)

        status_line = (
            "\u2705 Model tersedia"
            if model_in_memory
            else "\u26a0\ufe0f  Model tidak ada di disk"
        )
        metrics_text = f"Model: {model_name}\n{status_line}\n\n"

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
                metrics_text += (
                    f"Lowest Outlier Score: {src.get('lowest_outlier_score', 0):.4f}\n"
                )
                metrics_text += f"Highest Outlier Score: {src.get('highest_outlier_score', 0):.4f}\n"
                metrics_text += f"Average Outlier Score: {src.get('average_outlier_score', 0):.4f}\n"
                metrics_text += (
                    f"Training Duration: {src.get('training_duration', 0)} sec\n"
                )
                metrics_text += f"Model Name: {src.get('model_name', '-')}\n"
                metrics_text += f"Model Config:\n{src.get('model_configuration', {})}\n"
                metrics_text += "\n========== INSIGHT SUMMARY ==========\n\n"
                for item in src.get("insights", ["No major unusual pattern detected."]):
                    metrics_text += f"\u2022 {item}\n"
                metrics_text += "\n========== RECOMMENDATIONS ==========\n\n"
                for item in src.get("recommendations", ["Model appears stable."]):
                    metrics_text += f"\u2022 {item}\n"

                # ── FEATURE STATISTICS SECTION ────────────────────────────
                metrics_text += self._format_feature_statistics(feature_statistics)

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
                    metrics_text += top_outliers[available_cols].to_string(index=False)
                else:
                    metrics_text += "(Scored dataset belum dimuat)\n"
            else:
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_text += f"{metric}: {value:.4f}\n"
                    else:
                        metrics_text += f"{metric}: {value}\n"

                # ── FEATURE STATISTICS SECTION (non-anomaly) ──────────────
                metrics_text += self._format_feature_statistics(feature_statistics)

        self.metrics_content_label.configure(state="normal")
        self.metrics_content_label.delete("1.0", "end")
        self.metrics_content_label.insert("1.0", metrics_text)
        self.metrics_content_label.configure(state="disabled")

    # =========================================================
    # HELPER: Load feature_statistics dari joblib
    # =========================================================

    def _load_feature_statistics(self, model_name: str) -> dict:
        """
        Load feature_statistics dari file .joblib model.
        Return dict kosong jika tidak tersedia.
        """
        import os
        import joblib

        try:
            project_name = getattr(self.app, "current_project", None)
            if not project_name:
                return {}

            model_path = os.path.join(
                "projects",
                project_name,
                "artifacts",
                f"{model_name}_selected_model.joblib",
            )

            if not os.path.exists(model_path):
                return {}

            loaded = joblib.load(model_path)
            return loaded.get("feature_statistics", {})

        except Exception as e:
            print(f"[_load_feature_statistics] Gagal load: {e}")
            return {}

    # =========================================================
    # HELPER: Format feature_statistics jadi text display
    # =========================================================

    def _format_feature_statistics(self, feature_statistics: dict) -> str:

        if not feature_statistics:
            return (
                "\n========== FEATURE STATISTICS ==========\n\n"
                "(Tidak tersedia — model lama, re-train untuk generate statistik)\n"
            )

        lines = []
        lines.append(
            "\n========== FEATURE STATISTICS (dari Training Data) ==========\n"
        )
        lines.append("Batas normal = mean ± 2×std  |  nilai negatif di-clamp ke 0\n\n")

        col_feature = 30
        col_mean = 18
        col_std = 16
        col_min = 16
        col_max = 16

        header = (
            f"{'Feature':<{col_feature}}"
            f"{'Mean':>{col_mean}}"
            f"{'Std':>{col_std}}"
            f"{'Normal Min':>{col_min}}"
            f"{'Normal Max':>{col_max}}"
        )
        separator = "-" * (col_feature + col_mean + col_std + col_min + col_max)

        lines.append(header)
        lines.append(separator)

        def _fmt(val: float) -> str:
            """Format angka besar jadi readable: 1,234,567.89 atau scientific jika sangat besar."""
            abs_val = abs(val)
            if abs_val == 0:
                return "0.00"
            elif abs_val >= 1_000_000_000:
                return f"{val/1_000_000_000:.2f}B"
            elif abs_val >= 1_000_000:
                return f"{val/1_000_000:.2f}M"
            elif abs_val >= 1_000:
                return f"{val/1_000:.2f}K"
            elif abs_val >= 1:
                return f"{val:,.4f}"
            else:
                return f"{val:.6f}"

        for feature, stats in feature_statistics.items():
            mean = stats.get("mean", 0)
            std = stats.get("std", 0)
            normal_min = max(0, stats.get("normal_min", 0))  # clamp negatif ke 0
            normal_max = stats.get("normal_max", 0)

            row = (
                f"{feature:<{col_feature}}"
                f"{_fmt(mean):>{col_mean}}"
                f"{_fmt(std):>{col_std}}"
                f"{_fmt(normal_min):>{col_min}}"
                f"{_fmt(normal_max):>{col_max}}"
            )
            lines.append(row)

        lines.append(separator)
        lines.append(f"Total features: {len(feature_statistics)}\n")
        lines.append("Satuan: B=Miliar, M=Juta, K=Ribu\n")

        return "\n".join(lines) + "\n"

    def _confirm_model_selection(self):
        if not self.selected_model_name_for_display:
            messagebox.showwarning(
                "Pilih Model", "Klik model dari daftar terlebih dahulu."
            )
            return

        all_trained = getattr(self.app, "all_trained_models", {})
        model_pipeline = all_trained.get(self.selected_model_name_for_display)

        if model_pipeline is None:
            messagebox.showerror(
                "Model Tidak Tersedia",
                f"Model \u2019{self.selected_model_name_for_display}\u2019 tidak ada di disk.\n\n"
                "Kemungkinan penyebab:\n"
                "\u2022 Project dibuat sebelum fitur multi-model save ditambahkan\n"
                "\u2022 File .joblib terhapus\n\n"
                "Solusi: Kembali ke Configure dan jalankan ulang training.",
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

        if (
            hasattr(self, "confirm_status_label")
            and self.confirm_status_label.winfo_exists()
        ):
            self.confirm_status_label.configure(
                text=f"\u2705 \u2019{self.selected_model_name_for_display}\u2019 aktif sebagai model utama",
                text_color="#2ecc71",
            )

        messagebox.showinfo(
            "Model Dikonfirmasi",
            f"\u2705 Model \u2019{self.selected_model_name_for_display}\u2019 berhasil dipilih!\n"
            "Lanjut ke Try Model untuk mencoba prediksi.",
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 5 — TRY MODEL
    # ─────────────────────────────────────────────────────────────────────────
