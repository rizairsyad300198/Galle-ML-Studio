"""
ui/pages/export_page.py
Step 6 — Export Model: copy .joblib & generate Python example.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import os

from ui.widgets import BasePage


class ExportPage(BasePage):
    def show_export_model_screen(self):
        self.clear_content()
        self.ws.sidebar_steps["export_model"].configure(state="normal")

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
                back_cmd=lambda: self._go_to_main_step(self.ws.MAIN_STEP_TRY_MODEL),
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
            back_cmd=lambda: self._go_to_main_step(self.ws.MAIN_STEP_TRY_MODEL),
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

        # ── guard checks (unchanged) ──────────────────────────────────────────
        if (
            not hasattr(self.app, "selected_best_model_path")
            or not self.app.selected_best_model_path
        ):
            from tkinter import messagebox

            messagebox.showerror("Error", "No selected model path available.")
            return

        if not hasattr(self.app, "selected_features") or not self.app.selected_features:
            from tkinter import messagebox

            messagebox.showerror("Error", "Selected features not available.")
            return

        if not hasattr(self.app, "inferred_task") or not self.app.inferred_task:
            from tkinter import messagebox

            messagebox.showerror("Error", "Inferred task type not available.")
            return

        import pandas as pd

        task = self.app.inferred_task
        features = self.app.selected_features
        model_path = self.app.selected_best_model_path

        # ── AML-specific sensible defaults ───────────────────────────────────
        normal_defaults = {
            "txn_income_ratio": 0.08,
            "fee_ratio": 0.01,
            "portfolio_impact": 0.05,
            "investor_age_days": 1200,
            "days_since_last_txn": 14,
            "txn_count_sid": 18,
            "rolling_avg_txn": 12_000_000,
            "rolling_std_txn": 2_500_000,
            "txn_velocity_score": 0.75,
            "txn_acceleration": 1.05,
            "behavioral_volatility": 0.80,
            "txn_gap_zscore": 0.40,
            "rapid_txn": 0,
            "txn_month": 8,
            "txn_dayofweek": 2,
            "txn_is_month_end": 0,
        }

        # ── build sample input dict lines ────────────────────────────────────
        input_lines = []
        for feature in features:
            if feature in normal_defaults:
                val = repr(normal_defaults[feature])
            elif pd.api.types.is_numeric_dtype(self.app.df[feature].dtype):
                val = "0.0"
            elif self.app.df[feature].dtype in ("object", "category"):
                first = (
                    repr(str(self.app.df[feature].dropna().iloc[0]))
                    if not self.app.df[feature].dropna().empty
                    else "'example_category'"
                )
                val = first
            else:
                val = "'example_value'"
            input_lines.append(f'    "{feature}": {val},')

        sample_input_block = "\n".join(input_lines)

        # ── task-specific prediction block ───────────────────────────────────
        if task == "anomaly":
            prediction_block = """\
        # ── predict ──────────────────────────────────────────────────────────
        raw_label     = model.predict(input_df)[0]           # 1 = normal, -1 = outlier
        anomaly_score = model.decision_function(input_df)[0] # lower = more suspicious
    
        prediction_result = "outlier" if raw_label == -1 else "normal"
    
        # ── severity ─────────────────────────────────────────────────────────
        score = float(anomaly_score)
        if raw_label == 1:
            severity = "normal"
        elif score < -0.22:
            severity = "critical"
        elif score < -0.18:
            severity = "high"
        elif score < -0.12:
            severity = "medium"
        else:
            severity = "low"
    
        response = {
            "prediction":    prediction_result,
            "anomaly_score": round(score, 6),
            "severity":      severity,
            "raw_label":     int(raw_label),
        }"""

        elif task == "classification":
            prediction_block = """\
        # ── predict ──────────────────────────────────────────────────────────
        raw_pred = model.predict(input_df)[0]
    
        if label_encoder is not None:
            prediction_result = label_encoder.inverse_transform([int(raw_pred)])[0]
        else:
            prediction_result = raw_pred
    
        response = {
            "prediction": str(prediction_result),
            "raw_label":  int(raw_pred),
        }"""

        else:  # regression
            prediction_block = """\
        # ── predict ──────────────────────────────────────────────────────────
        predicted_value = float(model.predict(input_df)[0])
    
        response = {
            "prediction": round(predicted_value, 4),
        }"""

        # ── assemble full example code ────────────────────────────────────────
        example_code = f'''\
    import joblib
    import pandas as pd
    
    # =====================================================
    # MODEL PATH  — update sesuai lokasi file .joblib kamu
    # =====================================================
    MODEL_PATH = r"{model_path}"
    
    # =====================================================
    # LAZY LOAD  — load sekali, reuse setiap request
    # =====================================================
    _pipeline      = None
    _label_encoder = None
    
    
    def load_model():
        global _pipeline, _label_encoder
        if _pipeline is None:
            loaded      = joblib.load(MODEL_PATH)
            _pipeline   = loaded["pipeline"]
            _label_encoder = loaded.get("label_encoder")
        return _pipeline, _label_encoder
    
    
    # =====================================================
    # INFERENCE FUNCTION
    # =====================================================
    def run_inference(raw_data: dict) -> dict:
        """
        Parameters
        ----------
        raw_data : dict
            Feature dict sesuai kolom yang dipakai saat training.
            Contoh: {{"txn_income_ratio": 0.08, "fee_ratio": 0.01, ...}}
    
        Returns
        -------
        dict  dengan key: prediction, anomaly_score, severity, raw_label
        """
        model, label_encoder = load_model()
    
        input_df = pd.DataFrame([raw_data])
    
    {prediction_block}
    
        return response
    
    
    # =====================================================
    # CONTOH PEMAKAIAN
    # =====================================================
    if __name__ == "__main__":
    
        sample_input = {{
    {sample_input_block}
        }}
    
        result = run_inference(sample_input)
    
        print("Prediction    :", result["prediction"])
        print("Anomaly Score :", result.get("anomaly_score", "-"))
        print("Severity      :", result.get("severity", "-"))
        print("Full response :", result)
    '''

        # ── show popup (unchanged UI pattern) ─────────────────────────────────
        import customtkinter as ctk

        code_window = ctk.CTkToplevel(self.app)
        code_window.title("Python Usage Example")
        code_window.geometry("900x700")
        code_window.grab_set()

        textbox = ctk.CTkTextbox(
            code_window,
            wrap="word",
            font=("Consolas", 12),
            border_spacing=12,
        )
        textbox.pack(fill="both", expand=True, padx=10, pady=10)
        textbox.insert("1.0", example_code)
        textbox.configure(state="disabled")

        # ── copy to clipboard button ───────────────────────────────────────────
        def copy_to_clipboard():
            code_window.clipboard_clear()
            code_window.clipboard_append(example_code)
            code_window.update()
            copy_btn.configure(text="✅  Copied!")
            code_window.after(2000, lambda: copy_btn.configure(text="📋  Copy Code"))

        btn_frame = ctk.CTkFrame(code_window, fg_color="transparent")
        btn_frame.pack(pady=(0, 10))

        copy_btn = ctk.CTkButton(
            btn_frame,
            text="📋  Copy Code",
            width=160,
            fg_color="#3498db",
            hover_color="#2980b9",
            command=copy_to_clipboard,
        )
        copy_btn.grid(row=0, column=0, padx=8)

        ctk.CTkButton(
            btn_frame,
            text="Tutup",
            width=120,
            fg_color="gray40",
            hover_color="gray30",
            command=code_window.destroy,
        ).grid(row=0, column=1, padx=8)

    # ─────────────────────────────────────────────────────────────────────────
    #  METRICS POPUP
    # ─────────────────────────────────────────────────────────────────────────
