"""
ui/pages/try_model_page.py
Step 5 — Try Model: input fitur & dapatkan prediksi.
"""

import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
import numpy as np

from ui.widgets import BasePage


class TryModelPage(BasePage):
    def show_try_model_screen(self):
        self.clear_content()
        self.ws.sidebar_steps["try_model"].configure(state="normal")

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
                back_cmd=lambda: self._go_to_main_step(self.ws.MAIN_STEP_RESULTS),
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
            back_cmd=lambda: self._go_to_main_step(self.ws.MAIN_STEP_RESULTS),
            next_label="Next: Export Model →",
            next_cmd=lambda: self._go_to_main_step(self.ws.MAIN_STEP_EXPORT_MODEL),
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
