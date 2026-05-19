"""
ui/pages/configure_page.py
Step 2 — Configure Training sub-wizard:
  Target → Features → Models → Split & Launch
"""

import customtkinter as ctk
from tkinter import messagebox
import pandas as pd

from core.trainer import get_available_models, ALL_MODELS_CONFIG
from ui.widgets import BasePage


class ConfigurePage(BasePage):

    def __init__(self, ws):
        super().__init__(ws)
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
            back_cmd=lambda: self._go_to_main_step(self.ws.MAIN_STEP_TASK_SELECTION),
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

        # ── Cek kolom tanggal ────────────────────────────────────────────────────
        if "date" in target_col.lower() or "tgl" in target_col.lower():
            try:
                pd.to_datetime(target_data, errors="raise")
                is_suitable = False
                warning_text = "⚠️ Kolom tanggal tidak cocok sebagai target."
            except (ValueError, TypeError):
                pass

        # ── Cek kardinalitas terlalu tinggi ──────────────────────────────────────
        if is_suitable and not pd.api.types.is_numeric_dtype(target_data):
            if target_data.nunique() / len(target_data) > 0.5:
                is_suitable = False
                warning_text = "⚠️ Terlalu banyak nilai unik — kolom ini mungkin tidak cocok sebagai target."

        # ── Cek klasifikasi: minimal 2 class ─────────────────────────────────────
        if is_suitable:
            n_unique = target_data.nunique()
            task = getattr(self.app, "inferred_task", "")

            # Deteksi task jika belum di-set
            if task not in ("classification", "regression"):
                if pd.api.types.is_numeric_dtype(target_data) and n_unique > 20:
                    task = "regression"
                else:
                    task = "classification"

            if task == "classification" and n_unique < 2:
                is_suitable = False
                warning_text = (
                    f"❌ Kolom '{target_col}' hanya memiliki {n_unique} nilai unik.\n"
                    f"   Klasifikasi membutuhkan minimal 2 kelas."
                )

        # ── Update warning label ──────────────────────────────────────────────────
        if (
            hasattr(self, "target_warning_label")
            and self.target_warning_label.winfo_exists()
        ):
            self.target_warning_label.configure(
                text=warning_text if not is_suitable else "",
                text_color="#E74C3C" if warning_text.startswith("❌") else "#f39c12",
            )

        # ── Enable/disable tombol Next ────────────────────────────────────────────
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
            self._go_to_main_step(self.ws.MAIN_STEP_TASK_SELECTION)
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
        self.app.run_training()
        self._go_to_main_step(self.ws.MAIN_STEP_TRAIN)
