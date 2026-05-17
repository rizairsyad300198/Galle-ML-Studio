"""
ui/pages/metrics_popup.py
Popup penjelasan metrik evaluasi.
"""

import customtkinter as ctk
from ui.widgets import BasePage


class MetricsPopupMixin(BasePage):
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
