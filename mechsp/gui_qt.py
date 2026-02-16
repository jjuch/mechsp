# mechsp/gui_qt.py
import numpy as np

from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg

from .synthesis.synthesis import (make_quadratic_desired_field,
                        mix_with_circulation,
                        synthesize_currents,
                        synthesize_currents_weighted,
                        synthesize_currents_bounded)
from .dynamics import force_from_currents, simulate_second_order_ivp, simulate_second_order_chunked
from .interp import precompute_force_grid, make_force_interpolator

class SimWorker(QtCore.QObject):
    progress = QtCore.Signal(float) # 0..1
    finished = QtCore.Signal(object, object, object) # T, Q, V
    failed = QtCore.Signal(str)

    def __init__(self, q0, q_goal, L, coil_xy, I, h, mass, damping, t_final,
                 max_step, k_edge, margin, p_edge, marble_moment=1.0, scale=1.0, chunk_horizon=0.3):
        super().__init__()
        self.q0 = q0; self.q_goal = q_goal
        self.L = L; self.coil_xy = coil_xy; self.I = I; self.h = h
        self.mass = mass; self.damping = damping
        self.t_final = t_final; self.max_step = max_step
        self.k_edge = k_edge; self.margin = margin; self.p_edge = p_edge
        self.marble_moment = marble_moment; self.scale = scale
        self.chunk_horizon = chunk_horizon

    def run(self):
        try:
            # define force_fn in closure
            def force_fn(q):
                return force_from_currents(q, self.coil_xy, self.h, self.I, marble_moment=self.marble_moment, scale=self.scale)
            
            # progress callback
            def pcb(p):
                self.progress.emit(p)

            T, Q, V = simulate_second_order_chunked(self.q0, np.zeros(2), self.t_final, force_fn, self.L,
                                mass=self.mass, damping=self.damping,
                                k_edge=self.k_edge, margin=self.margin, p_edge=self.p_edge,
                                goal=self.q_goal, max_step=self.max_step,
                                chunk_horizon=self.chunk_horizon,
                                progress_cb=pcb
                )
            self.finished.emit(T, Q, V)
        except Exception as e:
            self.failed.emit(str(e))



class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- defaults ---
        self.L = 0.20
        self.h = 0.02
        self.k = 10.0
        self.mass = 1.0
        self.damping = 0.15
        self.t_final = 15.0
        self.max_step = 0.01
        self.lam = 1e-4
        self.marble_moment = 1.0
        self.scale = 1.0
        self.Gside = 80
        self.k_edge = 5.0
        self.margin = 0.02
        self.p_edge = 2

        self.q0 = np.array([-0.07, 0.06])
        self.q_goal = np.array([0.06, -0.03])
        self.n, self.m = 20, 20
        self.x_mix = 0.3
        self.solver_kind = "weighted"  # ridge|weighted|bounded
        self.sigma = 0.05
        self.Imax = 1.0

        self.coil_xy = None
        self.sample_xy = None
        self.I = None
        self.Q = None
        self.T = None

        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)

        # LEFT: plots stacked in a vertical layout
        left = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left)
        self.plot_field = pg.PlotWidget()
        self.plot_curr = pg.ImageView(view=pg.PlotItem())
        self.plot_dist = pg.PlotWidget()
        
        
        self.plot_field.setAspectLocked(True)
        self.plot_field.setRange(xRange=(-self.L/2, self.L/2), yRange=(-self.L/2, self.L/2))
        self.plot_field.showGrid(x=True, y=True)
        self.field_img = pg.ImageItem()
        self.plot_field.addItem(self.field_img)
        self.traj_curve = self.plot_field.plot(pen=pg.mkPen('r', width=2))

        # Add start and end marker
        self.start_marker = pg.ScatterPlotItem(
            size=12, symbol='t', brush=pg.mkBrush('g'), pen=pg.mkPen('k', width=1) # Green fill, black edge
        )
        self.goal_marker = pg.ScatterPlotItem(
            size=14, symbol='x', brush=pg.mkBrush('b'), pen=pg.mkPen('k', width=1) # Blue fill, black edge
        )
        self.plot_field.addItem(self.start_marker)
        self.plot_field.addItem(self.goal_marker)

        self.plot_dist.addLegend()
        self.cur_dx = self.plot_dist.plot(pen=pg.mkPen('c', width=1.5), name='dx')
        self.cur_dy = self.plot_dist.plot(pen=pg.mkPen('m', width=1.5), name='dy')
        self.cur_dn = self.plot_dist.plot(pen=pg.mkPen('y', width=2), name='||d||')


        left_v.addWidget(self.plot_field, 4)  # weight
        left_v.addWidget(self.plot_curr, 3)
        left_v.addWidget(self.plot_dist, 2)

        # Add progress bar
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        left_v.addWidget(self.progress)

        # RIGHT: compact form (inputs)
        right = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(right)
        
        
        self.ed_q0 = QtWidgets.QLineEdit(f"{self.q0[0]:.3f},{self.q0[1]:.3f}")
        self.ed_goal = QtWidgets.QLineEdit(f"{self.q_goal[0]:.3f},{self.q_goal[1]:.3f}")
        self.ed_grid = QtWidgets.QLineEdit(f"{self.n},{self.m}")
        self.ed_mix  = QtWidgets.QLineEdit(f"{self.x_mix:.2f}")
        self.cb_solver = QtWidgets.QComboBox()
        self.cb_solver.setToolTip("Choose synthesis solver:\n- ridge: unweighted ridge regression\n- weighted: goal-focused weighted fit\n- bounded: ridge with |I| ≤ Imax")
        self.cb_solver.addItems(["ridge", "weighted", "bounded"])
        self.cb_solver.setCurrentText("weighted") # default
        self.ed_sigma  = QtWidgets.QLineEdit(f"{self.sigma:.3f}")
        self.ed_imax   = QtWidgets.QLineEdit(f"{self.Imax:.2f}")
        self.ed_lam    = QtWidgets.QLineEdit(f"{self.lam:.1e}")
        self.ed_damp   = QtWidgets.QLineEdit(f"{self.damping:.2f}")

       
        form.addRow("q0 (x,y):", self.ed_q0)
        form.addRow("goal (x,y):", self.ed_goal)
        form.addRow("grid n,m:", self.ed_grid)
        form.addRow("mix x [0..1]:", self.ed_mix)
        form.addRow("solver:", self.cb_solver)
        form.addRow("sigma (weighted):", self.ed_sigma)
        form.addRow("Imax (bounded):", self.ed_imax)
        form.addRow("lambda (ridge):", self.ed_lam)
        form.addRow("damping:", self.ed_damp)
        self.btn_compute = QtWidgets.QPushButton("Compute")
        self.btn_compute.clicked.connect(self.on_compute)
        form.addRow(self.btn_compute)

        # Splitter (4/6 : 2/6)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 4)  # left weights
        split.setStretchFactor(1, 2)  # right weights

        root.addWidget(split)
        print("Build UI done.")

    def on_compute(self):
        print("Start computing")
        # parse
        try:
            self.q0 = np.array([float(s) for s in self.ed_q0.text().split(',')])
            self.q_goal = np.array([float(s) for s in self.ed_goal.text().split(',')])
            nm = [int(s) for s in self.ed_grid.text().split(',')]
            self.n, self.m = nm[0], nm[1]
            self.x_mix = float(self.ed_mix.text())
            self.solver_kind = self.cb_solver.currentText().lower()
            self.sigma = float(self.ed_sigma.text())
            self.Imax = float(self.ed_imax.text())
            self.lam = float(self.ed_lam.text())
            self.damping = float(self.ed_damp.text())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Parse error", str(e))
            return
        
        # Disable UI action while computing
        self.btn_compute.setEnabled(False)
        self.progress.setValue(0)

        # 1) coils
        self.coil_xy = self._make_coils()
        print("Coils made...")

        # 2) samples & target field
        sample_xy, F_target = self._make_target_field()
        print("Target fields made...")

        # 3) synthesis (selected solver)
        if self.solver_kind == "weighted":
            I, diag = synthesize_currents_weighted(
                sample_xy, F_target, self.coil_xy, self.h,
                q_goal=self.q_goal, sigma=self.sigma,
                lam=self.lam, marble_moment=self.marble_moment, scale=self.scale
            )
            print(f"[weighted] cond≈{diag['cond']:.2e}, rel_fit_w≈{diag['rel_fit_err_w']:.3e}")
        elif self.solver_kind == "bounded":
            I, diag = synthesize_currents_bounded(
                sample_xy, F_target, self.coil_xy, self.h,
                lam=self.lam, Imax=self.Imax,
                marble_moment=self.marble_moment, scale=self.scale
            )
            print(f"[bounded] rel_fit≈{diag['rel_fit_err']:.3e} ({diag.get('method','')})")
        else:
            I, diag = synthesize_currents(
                sample_xy, F_target, self.coil_xy, self.h,
                lam=self.lam, marble_moment=self.marble_moment, scale=self.scale
            )
            print(f"[ridge] cond≈{diag['cond']:.2e}, rel_fit≈{diag['rel_fit_err']:.3e}")

        self.I = I

        # 4) Launch simulation in background thread
        self.thread = QtCore.QThread(self)
        self.worker = SimWorker(
            q0=self.q0, q_goal=self.q_goal, L=self.L,
            coil_xy=self.coil_xy, I=self.I, h=self.h,
            mass=self.mass, damping=self.damping,
            t_final=self.t_final, max_step=self.max_step,
            k_edge=self.k_edge, margin=self.margin, p_edge=self.p_edge,
            marble_moment=self.marble_moment, scale=self.scale,
            chunk_horizon=0.3
        )
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        # cleanup
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.failed.connect(self.thread.quit)
        self.worker.failed.connect(self.worker.deleteLater)

        self.thread.start()
        print("Thread started...")
          
    

    def _on_progress(self, p):
        print(f"progress: {p} -> {int(100 * max(0.0, min(1.0, p)))}")
        self.progress.setValue(int(100 * max(0.0, min(1.0, p))))

    def _on_finished(self, T, Q, V):
        print("Thread finished...")
        self.T, self.Q = T, Q
        # update field image (coarse, fast)
        gx, gy, U, Vg = precompute_force_grid(self.coil_xy, self.h, self.I,
                        self.L, Gside=56,  # coarse
                        marble_moment=self.marble_moment, scale=self.scale)
        print("Force Grid precomputed...")
        
        self._update_plots(U, Vg, gx, gy)
        print("Plots updated...")
        self.progress.setValue(100)
        self.btn_compute.setEnabled(True)

    def _on_failed(self, msg):
        QtWidgets.QMessageBox.critical(self, "Simulation error: ", msg)
        self.btn_compute.setEnabled(True)
        self.progress.setValue(0)


    def _make_coils(self):
        d_x = self.L / (self.n + 1)
        d_y = self.L / (self.m + 1)
        xs = (np.arange(self.n) + 1) * d_x - self.L / 2
        ys = (np.arange(self.m) + 1) * d_y - self.L / 2
        XX, YY = np.meshgrid(xs, ys, indexing='xy')
        return np.stack([XX.ravel(), YY.ravel()], axis=1)

    def _make_target_field(self, Jside=28):
        xs = np.linspace(-self.L/2, self.L/2, Jside)
        ys = np.linspace(-self.L/2, self.L/2, Jside)
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        sample_xy = np.stack([X.ravel(), Y.ravel()], axis=1)

        F_des = make_quadratic_desired_field(sample_xy, self.q_goal, k=self.k)
        F_target = mix_with_circulation(F_des, self.x_mix)
        return sample_xy, F_target

    def _update_plots(self, U, V, gx, gy):
        # vector field magnitude image (fast) + trajectory overlay
        M = np.sqrt(U**2 + V**2)
        # pyqtgraph expects (col-major) image with y first; we can pass M as is, with proper rect
        img = M
        # set transform so pixels map to world coords
        tr = pg.QtGui.QTransform()
        dx = gx[1] - gx[0]
        dy = gy[1] - gy[0]
        tr.translate(gx[0], gy[0])
        tr.scale(dx, dy)
        self.field_img.setImage(img.T, autoLevels=True)  # transpose to align axes
        self.field_img.setTransform(tr)

        if self.Q is not None:
            # Add path and start, goal markers
            self.traj_curve.setData(self.Q[:, 0], self.Q[:, 1])
            self.start_marker.setData([self.q0[0]], [self.q0[1]])
            self.goal_marker.setData([self.q_goal[0]], [self.q_goal[1]])


            # distance plot
            dxy = self.Q - self.q_goal[None, :]
            dx = dxy[:, 0]; dy = dxy[:, 1]
            dn = np.sqrt(dx*dx + dy*dy)
            self.cur_dx.setData(self.T, dx)
            self.cur_dy.setData(self.T, dy)
            self.cur_dn.setData(self.T, dn)

        # currents heatmap (m×n)
        Igrid = self.I.reshape(self.n, self.m).T
        self.plot_curr.setImage(Igrid, autoLevels=True)
        self.plot_curr.view.setTitle("Coil currents")
        self.plot_curr.view.setLabel('bottom', 'x-index (n)')
        self.plot_curr.view.setLabel('left', 'y-index (m)')


def run_qt():
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()