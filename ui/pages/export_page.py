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
