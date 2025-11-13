import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
import sys
import traceback

# Windowsの高DPI対応（にじみ/ぼやけ対策）
def _enable_permonitor_dpi_awareness():
    try:
        import ctypes
        # Windows 10 以降
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PER_MONITOR_AWARE
    except Exception:
        try:
            # 旧API（Windows 8.1以前）
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

def _set_tk_scaling(root):
    """Tkのスケーリングを実ディスプレイDPIから自動設定"""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        hdc = ctypes.windll.user32.GetDC(0)
        LOGPIXELSX = 88
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
        scaling = dpi / 72.0  # 1ポイントあたりのピクセル数
        root.tk.call('tk', 'scaling', scaling)
    except Exception:
        root.tk.call('tk', 'scaling', 1.25)

# ---- あなたの既存のモジュール ----
from frag_data_writing import frag_data_writing

class FragDataGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # DPI最適化
        _enable_permonitor_dpi_awareness()
        _set_tk_scaling(self)

        self.title("IK-Frag: generating nuclear data for inverse kinematic reactions in PHITS")
        self.geometry("820x520")  # 少し広めでフォントに余裕を

        # --- フォント（Segoe UIに統一）---
        base = "Segoe UI"
        tkfont.nametofont("TkDefaultFont").configure(family=base, size=10)
        tkfont.nametofont("TkTextFont").configure(family=base, size=10)
        tkfont.nametofont("TkMenuFont").configure(family=base, size=10)
        tkfont.nametofont("TkHeadingFont").configure(family=base, size=11, weight="bold")
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            style.theme_use("clam")
        for w in ("TLabel", "TButton", "TEntry", "TCombobox", "TLabelframe.Label"):
            style.configure(w, font=(base, 10))
        style.configure("Status.TLabel", font=(base, 10, "italic"))
        style.configure("Heading.TLabel", font=(base, 11, "bold"))

        # 変数定義
        self.input_path = tk.StringVar(value="")
        self.nuclear_data = tk.StringVar(value="JENDL")  # TENDL or ENDF
        self.output_frag_path = tk.StringVar(value="frag_data.dat")
        self.step_energy = tk.DoubleVar(value=0.01)
        self.step_degree = tk.DoubleVar(value=0.05)
        self.material = tk.StringVar(value="Li")
        self.energy_threshold = tk.DoubleVar(value=5e6)

        # Notebook
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Tab 1: IO + library ---
        tab_io = ttk.Frame(nb)
        nb.add(tab_io, text="Data Selection")

        frm_in = ttk.LabelFrame(tab_io, text="Nuclear data input")
        frm_in.pack(fill="x", padx=10, pady=10)
        ttk.Label(frm_in, text="File path of input data:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ent_in = ttk.Entry(frm_in, textvariable=self.input_path, width=60)
        ent_in.grid(row=0, column=1, sticky="we", padx=8, pady=8)
        ttk.Button(frm_in, text="Select", command=self.browse_input).grid(row=0, column=2, padx=8, pady=8)
        frm_in.columnconfigure(1, weight=1)

        frm_out = ttk.LabelFrame(tab_io, text="Output (frag_data)")
        frm_out.pack(fill="x", padx=10, pady=10)
        ttk.Label(frm_out, text="Output file name:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ent_out = ttk.Entry(frm_out, textvariable=self.output_frag_path, width=72)
        ent_out.grid(row=0, column=1, sticky="we", padx=8, pady=8)
        ttk.Button(frm_out, text="Select", command=self.browse_output).grid(row=0, column=2, padx=8, pady=8)
        frm_out.columnconfigure(1, weight=1)

        frm_nuc = ttk.LabelFrame(tab_io, text="Nuclear Data Library")
        frm_nuc.pack(fill="x", padx=10, pady=10)
        ttk.Label(frm_nuc, text="Nuclear Data Library:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Combobox(
            frm_nuc, textvariable=self.nuclear_data,
            values=("JENDL", "TENDL", "ENDF"),
            state="readonly", width=10
        ).grid(row=0, column=1, padx=8, pady=8)

        frm_run = ttk.Frame(tab_io)
        frm_run.pack(fill="x", padx=10, pady=10)
        self.btn_run = ttk.Button(frm_run, text="Generate", command=self.run_generation)
        self.btn_run.pack(side="left")
        self.progress = ttk.Progressbar(frm_run, mode="indeterminate", length=220)
        self.progress.pack(side="left", padx=10)
        self.status = ttk.Label(frm_run, text="Ready", style="Status.TLabel")
        self.status.pack(side="left", padx=12)

        # --- Tab 2: Settings ---
        tab_cfg = ttk.Frame(nb)
        nb.add(tab_cfg, text="Settings")

        frm_cfg = ttk.LabelFrame(tab_cfg, text="Steps")
        frm_cfg.pack(fill="x", padx=10, pady=10)
        ttk.Label(frm_cfg, text="Energy step ΔE [MeV]:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(frm_cfg, textvariable=self.step_energy, width=12).grid(row=0, column=1, padx=8, pady=8)
        ttk.Label(frm_cfg, text="Degree step Δθ [deg]:").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(frm_cfg, textvariable=self.step_degree, width=12).grid(row=1, column=1, padx=8, pady=8)
        ttk.Label(tab_cfg, text="※ Defaults: ΔE=0.01 MeV, Δθ=0.05 deg", style="Status.TLabel")\
            .pack(anchor="w", padx=12, pady=6)

        frm_mat = ttk.LabelFrame(tab_cfg, text="Projectile")
        frm_mat.pack(fill="x", padx=10, pady=10)
        ttk.Label(frm_mat, text="Projectile (Li or Be):").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Combobox(
            frm_mat, textvariable=self.material,
            values=("Li", "Be"), state="readonly", width=10
        ).grid(row=0, column=1, padx=8, pady=8)

        frm_thr = ttk.LabelFrame(tab_cfg, text="Maximum projectile energy (eV)")
        frm_thr.pack(fill="x", padx=10, pady=10)
        ttk.Label(frm_thr, text="Maximum projectile energy for\nprocessing frag data:")\
            .grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(frm_thr, textvariable=self.energy_threshold, width=16)\
            .grid(row=0, column=1, padx=8, pady=8)
        frm_settings_run = ttk.Frame(frm_thr)
        frm_settings_run.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=10)
        ttk.Button(frm_settings_run, text="Generate", command=self.run_generation)\
            .pack(side="left")

        # フッター
        sep = ttk.Separator(self, orient="horizontal")
        sep.pack(fill="x", pady=(8, 0))
        ttk.Label(self, text="IK Frag", anchor="e").pack(anchor="e", padx=12, pady=8)

    # --------- UI handler ----------
    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Select nuclear data",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if path:
            self.input_path.set(path)

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Output file name (frag_data)",
            defaultextension=".dat",
            initialfile=self.output_frag_path.get() or "frag_data.dat",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if path:
            self.output_frag_path.set(path)

    def _validate_inputs(self):
        try:
            if self.step_energy.get() <= 0 or self.step_degree.get() <= 0:
                raise ValueError("Steps must be positive.")
            if self.material.get() not in ("Li", "Be"):
                raise ValueError("Projectile must be 'Li' or 'Be'.")
            if self.energy_threshold.get() <= 0:
                raise ValueError("Maximum projectile energy must be positive.")
        except Exception as e:
            messagebox.showwarning("Warning", f"Invalid settings:\n{e}")
            return False
        return True

    def run_generation(self):
        in_path = self.input_path.get().strip()
        out_path = self.output_frag_path.get().strip()

        if not in_path:
            messagebox.showwarning("Warning", "Please select the input data.")
            return
        if not out_path:
            messagebox.showwarning("Warning", "Please name the output file.")
            return
        if not self._validate_inputs():
            return

        self.btn_run.config(state="disabled")
        self.progress.start(30)
        self.status.config(text="Running...")

        th = threading.Thread(target=self._worker, args=(in_path, out_path), daemon=True)
        th.start()

    def _worker(self, in_path, out_path):
        ok = True
        msg = ""
        try:
            writer = frag_data_writing()
            writer.step_energy = float(self.step_energy.get())
            writer.step_degree = float(self.step_degree.get())
            writer.material = self.material.get().strip()
            writer.nuclear_data = self.nuclear_data.get().strip()
            writer.threshold_energy = float(self.energy_threshold.get())

            writer.load_material()
            writer.load_XS(in_path)
            writer.load_DDX(in_path)
            writer.load_neutron_data()
            writer.DDX_arrangement()
            writer.write_to_file(filename=out_path)
            msg = f"Completed!\nOutput: {out_path}"
        except Exception as e:
            ok = False
            tb = traceback.format_exc()
            msg = f"Error!\n{e}\n\nTraceback:\n{tb}"

        self.after(0, self._on_done, ok, msg)

    def _on_done(self, ok: bool, msg: str):
        self.progress.stop()
        self.btn_run.config(state="normal")
        self.status.config(text="Ready" if ok else "Error")
        if ok:
            messagebox.showinfo("Completed", msg)
        else:
            messagebox.showerror("Error", msg)


if __name__ == "__main__":
    app = FragDataGUI()
    app.mainloop()
