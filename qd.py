from collections import defaultdict
from itertools import combinations
import numpy as np
from numba import njit
import pyamg
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh as speigsh, cg
import time

class QuantumDots:
    def __init__(self, grid_size, grid_step, m_mask,
                 me_list, mh_list, Ve_list, Vh_list, eps_r_list,
                 Norb_max=10, N_stretch=10, s_max=10, p=2):
        """初始化量子点系统（精简版）"""
        print("=" * 50)
        print("初始化量子点系统 ... ")
        t_init_start = time.time()

        # 输入校验
        if len(grid_size) != 3 or len(grid_step) != 3:
            raise ValueError("grid_size和grid_step必须为3元组")

        # 物理常数（III-V族量子点常用值）
        self.hbar2_2m0 = 38.09981989277326  # ħ²/(2m0) = 38.0998 meV·nm²
        self.e2_ep0 = 18095.128179727828  # e²/ε0 = 18095.128179727828 meV * nm

        # 网格参数（精简命名）
        self.Nx, self.Ny, self.Nz = grid_size
        self.dx, self.dy, self.dz = grid_step
        self.N_total = self.Nx * self.Ny * self.Nz
        self.dx2, self.dy2, self.dz2 = self.dx ** 2, self.dy ** 2, self.dz ** 2
        self.dV = self.dx * self.dy * self.dz  # 体积元（提前计算）

        # 材料参数映射（精简逻辑）
        t_material = time.time()
        self.mask = m_mask.astype(int)
        self._init_material_params(me_list, mh_list, Ve_list, Vh_list, eps_r_list)
        print(f" - 材料参数映射耗时：{time.time() - t_material:.2f}s")

        # 泊松矩阵+AMG预条件子（核心保留）
        t_poisson = time.time()
        self.N_stretch, self.s_max, self.p = N_stretch, s_max, p
        self.A_poisson = self._build_poisson_matrix(N_stretch=self.N_stretch, s_max=self.s_max, p=self.p)
        self.amg_solver = pyamg.ruge_stuben_solver(self.A_poisson)
        print(f" - 泊松矩阵+AMG预条件子耗时：{time.time() - t_poisson:.2f}s")

        #'''
        # 轨道+库仑矩阵（核心计算）
        t_orbit = time.time()
        self.Eeo, self.orb_e, self.Eho, self.orb_h, self.V_ee, self.V_hh, self.V_eh, self.Neo, self.Nho = \
            self._compute_orbitals_and_coulomb(Ve_list, Vh_list, Norb_max)
        print(f" - 轨道+库仑矩阵耗时：{time.time() - t_orbit:.2f}s")
        print(f"   - Vee: {self.V_ee.shape}")
        print(f"   - Vhh: {self.V_hh.shape}")
        print(f"   - Veh: {self.V_eh.shape}")
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    for i4 in range(2):
                        print(f"{[i1, i2, i3, i4]}: {self.V_ee[i1, i2, i3, i4].tolist(), self.V_hh[i1, i2, i3, i4].tolist(), self.V_eh[i1, i2, i3, i4].tolist()}")

        # 二次量子化哈密顿量（精简构建）
        t_quant = time.time()
        self.H_single, self.H_double = self._build_second_quant_hamiltonian()
        print(f" - 二次量子化哈密顿量耗时：{time.time() - t_quant:.2f}s")

        print(f"\n✅ 初始化完成，总耗时：{time.time() - t_init_start:.2f}s")
        print("=" * 50)
        #'''

    def _init_material_params(self, me_list, mh_list, Ve_list, Vh_list, eps_r_list):
        """精简的材料参数初始化"""
        self.me = np.zeros_like(self.mask, dtype=np.float64)
        self.mh = np.zeros_like(self.mask, dtype=np.float64)
        self.Ve = np.zeros_like(self.mask, dtype=np.float64)
        self.Vh = np.zeros_like(self.mask, dtype=np.float64)
        self.eps_r = np.zeros_like(self.mask, dtype=np.float64)

        for mat_id in np.unique(self.mask):
            if mat_id >= len(me_list) or mat_id >= len(mh_list):
                raise ValueError(f"材料编号{mat_id}超出参数列表长度")
            mask_idx = (self.mask == mat_id)
            self.me[mask_idx] = me_list[mat_id]
            self.mh[mask_idx] = mh_list[mat_id]
            self.Ve[mask_idx] = Ve_list[mat_id]
            self.Vh[mask_idx] = Vh_list[mat_id]
            self.eps_r[mask_idx] = eps_r_list[mat_id]

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _build_sparse_hamiltonian_numba(Nx, Ny, Nz, dx2, dy2, dz2, func_r, V_r, hbar2_2m0):
        """合并泊松/势能矩阵构建（减少重复代码）"""
        N_total = Nx * Ny * Nz
        row = np.empty(N_total * 7, dtype=np.int32)
        col = np.empty(N_total * 7, dtype=np.int32)
        data = np.empty(N_total * 7, dtype=np.float64)
        ptr = 0

        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    idx = i + j * Nx + k * Nx * Ny
                    # 泊松矩阵逻辑（-∇·(ε∇φ)）
                    func_ijk = func_r[i, j, k]

                    # 对角元
                    i_left = i - 1
                    i_right = i + 1
                    func_left = 0.5 * (func_ijk + func_r[i_left, j, k]) if i_left >= 0 else 0.5 * func_ijk
                    func_right = 0.5 * (func_ijk + func_r[i_right, j, k]) if i_right < Nx else 0.5 * func_ijk
                    kin_x = (func_left + func_right) / dx2

                    j_down = j - 1
                    j_up = j + 1
                    func_down = 0.5 * (func_ijk + func_r[i, j_down, k]) if j_down >= 0 else 0.5 * func_ijk
                    func_up = 0.5 * (func_ijk + func_r[i, j_up, k]) if j_up < Ny else 0.5 * func_ijk
                    kin_y = (func_down + func_up) / dy2

                    k_back = k - 1
                    k_forward = k + 1
                    func_back = 0.5 * (func_ijk + func_r[i, j, k_back]) if k_back >= 0 else 0.5 * func_ijk
                    func_forward = 0.5 * (func_ijk + func_r[i, j, k_forward]) if k_forward < Nz else 0.5 * func_ijk
                    kin_z = (func_back + func_forward) / dz2

                    row[ptr] = idx
                    col[ptr] = idx
                    data[ptr] = (kin_x + kin_y + kin_z) * hbar2_2m0 + V_r[i, j, k]
                    if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1 or k == 0 or k == Nz - 1:
                        data[ptr] += 1e6
                    ptr += 1

                    # 非对角元
                    for di in [-1, 1]:
                        i_neigh = i + di
                        if 0 <= i_neigh < Nx:
                            idx_neigh = i_neigh + j * Nx + k * Nx * Ny
                            func_avg = (func_ijk + func_r[i_neigh, j, k]) / 2.0
                            row[ptr] = idx
                            col[ptr] = idx_neigh
                            data[ptr] = - func_avg / dx2 * hbar2_2m0
                            ptr += 1

                    for dj in [-1, 1]:
                        j_neigh = j + dj
                        if 0 <= j_neigh < Ny:
                            idx_neigh = i + j_neigh * Nx + k * Nx * Ny
                            func_avg = (func_ijk + func_r[i, j_neigh, k]) / 2.0
                            row[ptr] = idx
                            col[ptr] = idx_neigh
                            data[ptr] = - func_avg / dy2 * hbar2_2m0
                            ptr += 1

                    for dk in [-1, 1]:
                        k_neigh = k + dk
                        if 0 <= k_neigh < Nz:
                            idx_neigh = i + j * Nx + k_neigh * Nx * Ny
                            func_avg = (func_ijk + func_r[i, j, k_neigh]) / 2.0
                            row[ptr] = idx
                            col[ptr] = idx_neigh
                            data[ptr] = - func_avg / dz2 * hbar2_2m0
                            ptr += 1

        return row[:ptr], col[:ptr], data[:ptr]

    def _build_single_particle_hamiltonian(self, particle_type):
        """精简的单粒子哈密顿量构建"""
        if particle_type == 'electron':
            m_r = self.me
            V_r = self.Ve
        elif particle_type == 'hole':
            m_r = self.mh
            V_r = self.Vh
        else:
            raise ValueError("particle_type必须为'electron'或'hole'")
        if np.min(m_r) <= 0:
            raise ValueError("有效质量必须大于0")

        # 动能矩阵
        row, col, data = self._build_sparse_hamiltonian_numba(self.Nx, self.Ny, self.Nz,
                                                   self.dx2, self.dy2, self.dz2, 1 / m_r, V_r, self.hbar2_2m0)
        H_sparse = coo_matrix((data, (row, col)), shape=(self.N_total, self.N_total)).tocsr()

        # 对称化（保证厄米性）
        H_sparse = (H_sparse + H_sparse.T) / 2
        return H_sparse

    def _diagonalize_hamiltonian(self, particle_type, N_lowest=10):
        """精简的哈密顿量对角化"""
        H_sparse = self._build_single_particle_hamiltonian(particle_type)
        N_lowest = min(N_lowest, self.N_total)

        energies, eigenvectors = speigsh(H_sparse, k=N_lowest, which='SA', tol=1e-6,
                                         ncv=max(50, min(3 * N_lowest, self.N_total)))
        # 归一化（体积元）
        eigenvectors /= np.sqrt(self.dV)
        return energies, eigenvectors

    # ====================== 新增：PML拉伸系数计算（预处理，支持N_stretch层） ======================
    # ====================== 1. PML拉伸系数计算（明确PML在物理网格外） ======================
    @staticmethod
    @njit(fastmath=True, cache=True)
    def compute_stretch_coeff(N_dim, N_stretch, s_max=30.0, p=3):
        """
        计算单个方向的多项式拉伸系数（PML仅在内部物理网格外侧）
        参数：
            N_dim: 内部物理网格数（不包含PML层）
            N_stretch: 物理网格单侧的PML层数（外侧共2*N_stretch层PML）
            s_max: 最大拉伸系数（推荐20-50）
            p: 多项式阶数（推荐3，保证界面连续）
        返回：
            s: 长度为N_total的拉伸系数数组（内部物理网格s=1，外侧PML层s>1）
        """
        Nxt = N_stretch + N_dim + N_stretch
        s = np.ones(Nxt, dtype=np.float64)

        # 左侧PML层（0 ~ N_stretch-1）：物理网格左侧
        for i in range(N_stretch):
            t = (N_stretch - 1 - i) / N_stretch  # 归一化距离（0~1）
            s[i] = 1.0 + s_max * (t ** p)

        # 右侧PML层（N_total-N_stretch ~ N_total-1）：物理网格右侧
        start_idx = N_stretch + N_dim
        for i in range(start_idx, Nxt):
            t = (i - start_idx) / N_stretch  # 归一化距离（0~1）
            s[i] = 1.0 + s_max * (t ** p)

        # 中间区域（N_stretch ~ N_total-N_stretch-1）：内部有效物理网格，s=1
        return s

    # ====================== 2. 扩展介电常数计算 ======================
    @staticmethod
    def _extend_matrix_3d(matrix_3d, N_stretch, mode='constant'):
        # 扩展介电常数
        pad_width = ((N_stretch, N_stretch), (N_stretch, N_stretch), (N_stretch, N_stretch))
        if mode == 'constant':
            matrix_3d_full = np.pad(matrix_3d, pad_width, mode='constant')
        else:
            matrix_3d_full = np.pad(matrix_3d, pad_width, mode='edge')

        return matrix_3d_full

    # ====================== 3. PML各向异性介电常数计算 ======================
    @staticmethod
    def build_pml_epsilon(eps_scalar_full, s_x, s_y, s_z):
        # 向量化广播，一行完成三维张量计算，和原逻辑结果完全一致
        s_x_3d = s_x.reshape(-1, 1, 1)
        s_y_3d = s_y.reshape(1, -1, 1)
        s_z_3d = s_z.reshape(1, 1, -1)

        # PML各向异性介电常数张量公式，完全符合静电场PML理论
        eps_x = eps_scalar_full * (s_y_3d * s_z_3d) / s_x_3d
        eps_y = eps_scalar_full * (s_x_3d * s_z_3d) / s_y_3d
        eps_z = eps_scalar_full * (s_x_3d * s_y_3d) / s_z_3d

        return eps_x, eps_y, eps_z

    # ====================== 3. 带PML的稀疏矩阵构造（核心逻辑不变） ======================

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _build_poisson_sparse_matrix(dx2, dy2, dz2, eps_x, eps_y, eps_z):
        """
        重构后的稀疏矩阵构造函数 -∇·(ε∇φ), 适配PML各向异性介电常数，添加一个负号，后面泊松方程有面电荷密度不加负号。
        """
        Nxt, Nyt, Nzt = eps_x.shape
        N_total = Nxt * Nyt * Nzt
        row = np.empty(N_total * 7, dtype=np.int32)
        col = np.empty(N_total * 7, dtype=np.int32)
        data = np.empty(N_total * 7, dtype=np.float64)
        ptr = 0

        for k in range(Nzt):
            for j in range(Nyt):
                for i in range(Nxt):
                    idx = i + j * Nxt + k * Nxt * Nyt

                    # X方向（用eps_x）
                    eps_x_ijk = eps_x[i, j, k]
                    i_left = i - 1
                    i_right = i + 1
                    func_left = 0.5 * (eps_x_ijk + eps_x[i_left, j, k]) if i_left >= 0 else 0.5 * eps_x_ijk
                    func_right = 0.5 * (eps_x_ijk + eps_x[i_right, j, k]) if i_right < Nxt else 0.5 * eps_x_ijk
                    kin_x = (func_left + func_right) / dx2

                    # Y方向（用eps_y）
                    eps_y_ijk = eps_y[i, j, k]
                    j_down = j - 1
                    j_up = j + 1
                    func_down = 0.5 * (eps_y_ijk + eps_y[i, j_down, k]) if j_down >= 0 else 0.5 * eps_y_ijk
                    func_up = 0.5 * (eps_y_ijk + eps_y[i, j_up, k]) if j_up < Nyt else 0.5 * eps_y_ijk
                    kin_y = (func_down + func_up) / dy2

                    # Z方向（用eps_z）
                    eps_z_ijk = eps_z[i, j, k]
                    k_back = k - 1
                    k_forward = k + 1
                    func_back = 0.5 * (eps_z_ijk + eps_z[i, j, k_back]) if k_back >= 0 else 0.5 * eps_z_ijk
                    func_forward = 0.5 * (eps_z_ijk + eps_z[i, j, k_forward]) if k_forward < Nzt else 0.5 * eps_z_ijk
                    kin_z = (func_back + func_forward) / dz2

                    # 对角元赋值
                    row[ptr] = idx
                    col[ptr] = idx
                    data[ptr] = kin_x + kin_y + kin_z
                    ptr += 1

                    # X方向邻居（用eps_x）
                    for di in [-1, 1]:
                        i_neigh = i + di
                        if 0 <= i_neigh < Nxt:
                            idx_neigh = i_neigh + j * Nxt + k * Nxt * Nyt
                            func_avg = (eps_x_ijk + eps_x[i_neigh, j, k]) / 2.0
                            row[ptr] = idx
                            col[ptr] = idx_neigh
                            data[ptr] = - func_avg / dx2
                            ptr += 1

                    # Y方向邻居（用eps_y）
                    for dj in [-1, 1]:
                        j_neigh = j + dj
                        if 0 <= j_neigh < Nyt:
                            idx_neigh = i + j_neigh * Nxt + k * Nxt * Nyt
                            func_avg = (eps_y_ijk + eps_y[i, j_neigh, k]) / 2.0
                            row[ptr] = idx
                            col[ptr] = idx_neigh
                            data[ptr] = - func_avg / dy2
                            ptr += 1

                    # Z方向邻居（用eps_z）
                    for dk in [-1, 1]:
                        k_neigh = k + dk
                        if 0 <= k_neigh < Nzt:
                            idx_neigh = i + j * Nxt + k_neigh * Nxt * Nyt
                            func_avg = (eps_z_ijk + eps_z[i, j, k_neigh]) / 2.0
                            row[ptr] = idx
                            col[ptr] = idx_neigh
                            data[ptr] = - func_avg / dz2
                            ptr += 1

        return row[:ptr], col[:ptr], data[:ptr]

    def _build_poisson_matrix(self, N_stretch, s_max, p):
        """构建泊松方程矩阵"""
        s_x = self.compute_stretch_coeff(self.Nx, N_stretch, s_max, p)
        s_y = self.compute_stretch_coeff(self.Ny, N_stretch, s_max, p)
        s_z = self.compute_stretch_coeff(self.Nz, N_stretch, s_max, p)
        eps_full = self._extend_matrix_3d(self.eps_r, N_stretch, mode='edge')
        eps_x, eps_y, eps_z = self.build_pml_epsilon(eps_full, s_x, s_y, s_z)

        row, col, data = self._build_poisson_sparse_matrix(
            self.dx2, self.dy2, self.dz2, eps_x, eps_y, eps_z)
        N_pml_total = len(s_x) * len(s_y) * len(s_z)
        return coo_matrix((data, (row, col)), shape=(N_pml_total, N_pml_total)).tocsr()

    def solve_poisson(self, rho):
        """求解泊松方程（精简接口）"""
        rho_full = self._extend_matrix_3d(rho.reshape(self.Nx, self.Ny, self.Nz),
                                          N_stretch=self.N_stretch, mode='constant')

        phi, info = cg(self.A_poisson, rho_full.reshape((-1, )), M=self.amg_solver.aspreconditioner(),
                       rtol=1e-6, maxiter=1000)
        if info != 0:
            print(f"警告：CG迭代未收敛（info={info}）")
        phi = phi.reshape(self.Nx + 2*self.N_stretch, self.Ny + 2*self.N_stretch, self.Nz + 2*self.N_stretch)
        phi = phi[self.N_stretch:self.Nx + self.N_stretch,
                  self.N_stretch:self.Ny + self.N_stretch,
                  self.N_stretch:self.Nz + self.N_stretch]
        phi = phi.reshape((-1, ))
        return phi

    def _compute_charge_density_and_potential(self, orb_e, orb_h):
        """计算电荷密度和电势（精简逻辑）"""
        orb_e = orb_e.T  # (Neo, N_total)
        orb_h = orb_h.T  # (Nho, N_total)
        Neo, Nho = orb_e.shape[0], orb_h.shape[0]
        print("orb_e.shape, orb_h.shape:", orb_e.shape, orb_h.shape)

        # 电子电荷密度+电势
        rho_e = - np.einsum('i...,j...->ij...', orb_e, orb_e)
        phi_e = np.zeros_like(rho_e)
        for i in range(Neo):
            for j in range(i, Neo):
                phi_e[i, j] = self.solve_poisson(rho_e[i, j])
                if i != j:
                    phi_e[j, i] = phi_e[i, j]

        # 空穴电荷密度+电势
        rho_h = np.einsum('i...,j...->ij...', orb_h, orb_h)
        phi_h = np.zeros_like(rho_h)
        for i in range(Nho):
            for j in range(i, Nho):
                phi_h[i, j] = self.solve_poisson(rho_h[i, j])
                if i != j:
                    phi_h[j, i] = phi_h[i, j]

        '''
        phi_x = phi_e[0, 0].reshape(self.Nx, self.Ny, self.Nz)[:, self.Ny//2, self.Nz//2]
        import matplotlib.pyplot as plt
        x = np.linspace(-5, 5, self.Nx)
        plt.plot(x, phi_x)

        x = np.linspace(-5, 5, self.Nx)
        phy_x_theory = 1 / (4 * np.pi * 10 * np.abs(x + 1e-3))
        plt.plot(x, - phy_x_theory)
        plt.ylim([np.min(phi_x), np.max(phi_x)])
        plt.show()
        '''
        return rho_e, phi_e, rho_h, phi_h

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _compute_coulomb_matrix_numba(rho_e, phi_e, rho_h, phi_h, pdx_ee, pdx_hh, pdx_eh, dV, e2_ep0):
        """Numba加速的库仑矩阵计算（核心优化）"""
        Neo = rho_e.shape[0]
        Nho = rho_h.shape[0]

        # 电子-电子库仑相互作用
        V_ee = np.zeros((Neo, Neo, Neo, Neo))
        for i1, i2, i_dx in pdx_ee:
            if i1 > i2:
                continue
            for j1, j2, j_dx in pdx_ee[i_dx:]:
                val = np.sum(rho_e[i1, j1] * phi_e[i2, j2]) * dV * e2_ep0
                V_ee[i1, i2, j1, j2] = val
                V_ee[j1, j2, i1, i2] = val
                V_ee[i2, i1, j2, j1] = val
                V_ee[j2, j1, i2, i1] = val
        # 空穴-空穴库仑相互作用
        V_hh = np.zeros((Nho, Nho, Nho, Nho))
        for i1, i2, i_dx in pdx_hh:
            if i1 > i2:
                continue
            for j1, j2, j_dx in pdx_hh[i_dx:]:
                val = np.sum(rho_h[i1, j1] * phi_h[i2, j2]) * dV * e2_ep0
                V_hh[i1, i2, j1, j2] = val
                V_hh[j1, j2, i1, i2] = val
                V_hh[i2, i1, j2, j1] = val
                V_hh[j2, j1, i2, i1] = val
        # 电子-空穴库仑相互作用
        V_eh = np.zeros((Neo, Nho, Neo, Nho))
        for i1, i2, i_dx in pdx_eh:
            for j1, j2, j_dx in pdx_eh[i_dx:]:
                val = np.sum(rho_e[i1, j1] * phi_h[i2, j2]) * dV * e2_ep0
                V_eh[i1, i2, j1, j2] = val
                V_eh[j1, j2, i1, i2] = val

        return V_ee, V_hh, V_eh

    def _compute_orbitals_and_coulomb(self, Ve_list, Vh_list, Norb_max):
        """精简的轨道+库仑矩阵计算"""
        # 对角化得到轨道
        Eeo, orb_e = self._diagonalize_hamiltonian('electron', N_lowest=Norb_max)
        Eho, orb_h = self._diagonalize_hamiltonian('hole', N_lowest=Norb_max)

        # 筛选物理轨道（基于势垒）
        Ve_max = np.max(Ve_list) if len(Ve_list) <= 2 else np.max(Ve_list)  # np.max(Ve_list[:-1])
        Vh_max = np.max(Vh_list) if len(Vh_list) <= 2 else np.max(Vh_list)  # np.max(Vh_list[:-1])
        Neo, Nho = np.sum(Eeo < Ve_max), np.sum(Eho < Vh_max)
        Neo = 2 if Neo < 2 else Neo
        Nho = 2 if Nho < 2 else Nho

        Eeo, orb_e = Eeo[:Neo], orb_e[:, :Neo]
        Eho, orb_h = Eho[:Nho], orb_h[:, :Nho]

        # 电荷密度+电势
        rho_e, phi_e, rho_h, phi_h = self._compute_charge_density_and_potential(orb_e, orb_h)

        # 库仑矩阵（Numba加速）
        pdx_ee = np.column_stack((np.repeat(np.arange(Neo), Neo),
                                  np.tile(np.arange(Neo), Neo),
                                  np.arange(Neo*Neo)))
        pdx_hh = np.column_stack((np.repeat(np.arange(Nho), Nho),
                                  np.tile(np.arange(Nho), Nho),
                                  np.arange(Nho*Nho)))
        pdx_eh = np.column_stack((np.repeat(np.arange(Neo), Nho),
                                  np.tile(np.arange(Nho), Neo),
                                  np.arange(Neo*Nho)))
        V_ee, V_hh, V_eh = self._compute_coulomb_matrix_numba(rho_e, phi_e, rho_h, phi_h,
                                                              pdx_ee, pdx_hh, pdx_eh, self.dV, self.e2_ep0)
        return Eeo, orb_e, Eho, orb_h, V_ee, V_hh, V_eh, Neo, Nho

    def _build_second_quant_hamiltonian(self):
        t0 = time.time()
        n_orb = 2 * self.Neo + 2 * self.Nho
        single_particle_terms = np.zeros((n_orb), dtype=np.float64)  # 单粒子项
        # 1. 电子单粒子项
        for i in range(self.Neo):
            single_particle_terms[i] = self.Eeo[i]  # 电子↑
            single_particle_terms[i+self.Neo] = self.Eeo[i]  # 电子↓
        # 2. 空穴单粒子项
        offset_h = 2 * self.Neo  # 空穴轨道起始索引
        for j in range(self.Nho):
            single_particle_terms[offset_h + j] = self.Eho[j]  # 空穴↑
            single_particle_terms[offset_h + self.Nho + j] = self.Eho[j]  # 空穴↓

        static_double_particle_terms = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=np.float64)  # 相互作用项
        # 3. 电子-电子相互作用 (无自旋矩阵元)
        # 矩阵元: Vee_ijkl (i,j,k,l均为空间轨道指标)
        # 算符形式: 1/2 * Vee_ijkl * c^†_iσ c^†_jτ c_lτ c_kσ
        for i in range(self.Neo):
            for j in range(self.Neo):
                for k in range(self.Neo):
                    for l in range(self.Neo):
                        V = self.V_ee[i, j, k, l]
                        i_up, j_up = i, j  # 电子通道: ↑, ↑
                        k_up, l_up = k, l
                        i_dn, j_dn = i + self.Neo, j + self.Neo  # 电子通道: ↓, ↓
                        k_dn, l_dn = k + self.Neo, l + self.Neo
                        static_double_particle_terms[i_up, j_up, k_up, l_up] = 0.5 * V  # σ=↑, τ=↑
                        static_double_particle_terms[i_up, j_dn, k_up, l_dn] = 0.5 * V  # σ=↑, τ=↓
                        static_double_particle_terms[i_dn, j_up, k_dn, l_up] = 0.5 * V  # σ=↓, τ=↑
                        static_double_particle_terms[i_dn, j_dn, k_dn, l_dn] = 0.5 * V  # σ=↓, τ=↓
        # 4. 空穴-空穴相互作用 (无自旋矩阵元)
        for i in range(self.Nho):
            for j in range(self.Nho):
                for k in range(self.Nho):
                    for l in range(self.Nho):
                        V = self.V_hh[i, j, k, l]
                        i_up, j_up = offset_h + i, offset_h + j  # 空穴通道: ↑, ↑
                        k_up, l_up = offset_h + k, offset_h + l
                        i_dn, j_dn = offset_h + self.Nho + i, offset_h + self.Nho + j  # 空穴通道: ↓, ↓
                        k_dn, l_dn = offset_h + self.Nho + k, offset_h + self.Nho + l
                        static_double_particle_terms[i_up, j_up, k_up, l_up] = 0.5 * V  # σ=↑, τ=↑
                        static_double_particle_terms[i_up, j_dn, k_up, l_dn] = 0.5 * V  # σ=↑, τ=↓
                        static_double_particle_terms[i_dn, j_up, k_dn, l_up] = 0.5 * V  # σ=↓, τ=↑
                        static_double_particle_terms[i_dn, j_dn, k_dn, l_dn] = 0.5 * V  # σ=↓, τ=↓
        # 5. 电子-空穴相互作用
        # 算符: Veh_ijkl * e^+_i h^+_j h_l e_k
        # 矩阵元形状: (n_orb_e, n_orb_h, n_orb_e, n_orb_h)
        # 注意: V_{ijkl} 对应 e^+_i h^+_j h_l e_k
        for i in range(self.Neo):  # 电子轨道 i
            for j in range(self.Nho):  # 空穴轨道 j
                for k in range(self.Neo):  # 电子轨道 k
                    for l in range(self.Nho):  # 空穴轨道 l
                        V = self.V_eh[i, j, k, l]  # 注意顺序: i,j,k,l 对应 e^+_i h^+_j h_l e_k
                        # 自旋↑↑
                        i_e_up, j_h_up = i, offset_h + j
                        k_e_up, l_h_up = k, offset_h + l
                        # 自旋↓↓
                        i_e_dn, j_h_dn = i + self.Neo, offset_h + self.Nho + j
                        k_e_dn, l_h_dn = k + self.Neo, offset_h + self.Nho + l
                        static_double_particle_terms[i_e_up, j_h_up, k_e_up, l_h_up] = V  # σ=↑, τ=↑
                        static_double_particle_terms[i_e_up, j_h_dn, k_e_up, l_h_dn] = V  # σ=↑, τ=↓
                        static_double_particle_terms[i_e_dn, j_h_up, k_e_dn, l_h_up] = V  # σ=↓, τ=↑
                        static_double_particle_terms[i_e_dn, j_h_dn, k_e_dn, l_h_dn] = V  # σ=↓, τ=↓

        print("\n - 生成二次量子化 Hamiltonian 描述： 单粒子对角项，双粒子相互作用项")
        print(f" - 单粒子项长度：{single_particle_terms.shape}")
        print(f" - 双粒子项长度：{static_double_particle_terms.shape}")
        print(f" - 耗时：{time.time() - t0:.2f}秒")
        return  single_particle_terms, static_double_particle_terms

    def _generate_configs(self, ne_up, ne_dn, nh_up, nh_dn):
        """生成指定自旋占据数的组态（修复版）"""
        # 自旋轨道索引
        e_up = list(range(self.Neo))
        e_dn = list(range(self.Neo, 2 * self.Neo))
        h_up = list(range(2 * self.Neo, 2 * self.Neo + self.Nho))
        h_dn = list(range(2 * self.Neo + self.Nho, 2 * (self.Neo + self.Nho)))

        # 合法性校验
        if ne_up > len(e_up) or ne_dn > len(e_dn) or nh_up > len(h_up) or nh_dn > len(h_dn):
            raise ValueError("占据数超过轨道数")

        # 生成所有组合并转换为列表
        e_up_comb_list = list(combinations(e_up, ne_up))
        e_dn_comb_list = list(combinations(e_dn, ne_dn))
        h_up_comb_list = list(combinations(h_up, nh_up))
        h_dn_comb_list = list(combinations(h_dn, nh_dn))

        occ_lists = []
        for e_u in e_up_comb_list:
            for e_d in e_dn_comb_list:
                for h_u in h_up_comb_list:
                    for h_d in h_dn_comb_list:
                        occ = sorted(list(e_u) + list(e_d) + list(h_u) + list(h_d))
                        occ_lists.append((occ, len(occ_lists)))

        return occ_lists, 2 * (self.Neo + self.Nho)

    def _generate_biparticle_combinations(self, occ_lists):
        """生成双粒子组合（精简版）"""
        if len(occ_lists) == 0 or len(occ_lists[0][0]) <= 1:
            return []

        combs = []
        n_occ = len(occ_lists[0][0])
        idx_pairs = list(combinations(range(n_occ), 2))

        for cc, idx in occ_lists:
            cc_arr = np.array(cc)
            p_c_s = []
            for (i, j) in idx_pairs:
                a, b = cc[i], cc[j]
                # 计算符号
                sign = -((-1) ** (np.sum(a < cc_arr) + np.sum(b < cc_arr)))
                # 剩余轨道
                rest = sorted([cc[k] for k in range(n_occ) if k not in (i, j)])
                p_c_s.append([[a, b], rest, sign])
            combs.append([p_c_s, idx])

        return combs

    def _build_hamiltonian_matrix(self, ne_up, ne_dn, nh_up, nh_dn, use_sparse=True):
        """构建组态空间哈密顿量（修正版）"""
        # 生成组态（使用修复的_generate_configs）
        occ_lists, _ = self._generate_configs(ne_up, ne_dn, nh_up, nh_dn)
        n_cc = len(occ_lists)

        print(f" - 生成组态数：{n_cc}")

        if n_cc == 0:
            raise ValueError("无有效组态")

        # 1. 单粒子对角项
        if use_sparse:
            diag = np.zeros(n_cc)
            for occ, idx in occ_lists:
                diag[idx] = np.sum(self.H_single[occ])
            H0 = coo_matrix((diag, (range(n_cc), range(n_cc))), shape=(n_cc, n_cc)).tocsr()
        else:
            H0 = np.zeros((n_cc, n_cc))
            for occ, idx in occ_lists:
                H0[idx, idx] = np.sum(self.H_single[occ])

        # 2. 双粒子相互作用项（仅当总粒子数≥2时）
        neh = ne_up + ne_dn + nh_up + nh_dn
        if neh < 2:
            return H0, occ_lists

        # 生成双粒子组合
        combs = self._generate_biparticle_combinations(occ_lists)

        print(f" - 双粒子组合数：{len(combs)}")

        if use_sparse:
            # 使用稀疏矩阵构建
            rows, cols, data = [], [], []
            rest_groups = defaultdict(list)

            # 按剩余轨道分组
            for config_idx, (cc_rest_sign_list, _) in enumerate(combs):
                for cc, rest, sign in cc_rest_sign_list:
                    # 将剩余轨道转换为可哈希的键
                    if isinstance(rest, (list, np.ndarray)) and len(rest) > 0:
                        rest_key = tuple(sorted(rest))
                    else:
                        rest_key = tuple([-1])

                    rest_groups[rest_key].append({
                        'config_idx': config_idx,
                        'cc': cc,
                        'sign': sign
                    })

            # 计算矩阵元
            for rest_key, group in rest_groups.items():
                n_group = len(group)
                if n_group <= 1:
                    continue

                # 对组内的所有组合对进行计算
                for i in range(n_group):
                    comb_i = group[i]
                    config_i = comb_i['config_idx']
                    a1, b1 = comb_i['cc']
                    sign1 = comb_i['sign']

                    for j in range(i, n_group):
                        comb_j = group[j]
                        config_j = comb_j['config_idx']
                        a2, b2 = comb_j['cc']
                        sign2 = comb_j['sign']

                        # 计算矩阵元（原始物理公式）
                        sign_ = sign1 * sign2
                        term = (self.H_double[a1, b1, a2, b2] -
                                self.H_double[a1, b1, b2, a2] -
                                self.H_double[b1, a1, a2, b2] +
                                self.H_double[b1, a1, b2, a2]) * sign_

                        if abs(term) > 1e-15:
                            rows.append(config_i)
                            cols.append(config_j)
                            data.append(term)
                            if config_i != config_j:
                                rows.append(config_j)
                                cols.append(config_i)
                                data.append(term)

            H_int = coo_matrix((data, (rows, cols)), shape=(n_cc, n_cc)).tocsr()
            H_total = H0 + H_int

            # 统计信息
            nnz = H_total.nnz
            sparsity = 1.0 - (nnz / (n_cc * n_cc))
            print(f" - 哈密顿矩阵非零元：{nnz}，稀疏度：{sparsity:.4f}")

        else:
            # 密集矩阵构建
            H_int = np.zeros((n_cc, n_cc), dtype=np.float64)

            # 按剩余轨道分组
            rest_groups = defaultdict(list)

            # 构建剩余轨道到组合的映射
            for config_idx, (cc_rest_sign_list, _) in enumerate(combs):
                for cc, rest, sign in cc_rest_sign_list:
                    # 将剩余轨道转换为可哈希的键
                    if isinstance(rest, (list, np.ndarray)) and len(rest) > 0:
                        rest_key = tuple(sorted(rest))
                    else:
                        rest_key = tuple([-1])

                    rest_groups[rest_key].append({
                        'config_idx': config_idx,
                        'cc': cc,
                        'sign': sign
                    })

            # 计算矩阵元
            for rest_key, group in rest_groups.items():
                n_group = len(group)
                if n_group <= 1:
                    continue

                # 对组内的所有组合对进行计算
                for i in range(n_group):
                    comb_i = group[i]
                    config_i = comb_i['config_idx']
                    a1, b1 = comb_i['cc']
                    sign1 = comb_i['sign']

                    for j in range(i, n_group):
                        comb_j = group[j]
                        config_j = comb_j['config_idx']
                        a2, b2 = comb_j['cc']
                        sign2 = comb_j['sign']

                        # 计算矩阵元（原始物理公式）
                        sign_ = sign1 * sign2
                        term = (self.H_double[a1, b1, a2, b2] -
                                self.H_double[a1, b1, b2, a2] -
                                self.H_double[b1, a1, a2, b2] +
                                self.H_double[b1, a1, b2, a2]) * sign_

                        H_int[config_i, config_j] += term
                        if config_i != config_j:
                            H_int[config_j, config_i] += term

            H_total = H0 + H_int

            # 统计信息
            non_zero_elements = np.sum(np.abs(H_total) > 1e-15)
            total_elements = n_cc * n_cc
            sparsity = 1.0 - (non_zero_elements / total_elements)
            print(f" - 哈密顿矩阵非零元：{non_zero_elements}，稀疏度：{sparsity:.4f}")
        print(f" - 哈密顿矩阵厄米:", np.sum(np.abs(H_total - H_total.T.conj()))<1e-10)
        return H_total, occ_lists

    def get_ground_state(self, ne_up, ne_dn, nh_up, nh_dn, use_sparse=True):
        """获取基态能量（最终精简接口）"""
        t_start = time.time()
        H, occ_lists = self._build_hamiltonian_matrix(ne_up, ne_dn, nh_up, nh_dn, use_sparse)

        # 对角化
        if use_sparse:
            E, psi = speigsh(H, k=1, which='SA', tol=1e-8)
        else:
            from scipy.linalg import eigh
            E, psi = eigh(H, subset_by_index=(0, 0))

        print(f" - 基态计算完成，耗时：{time.time() - t_start:.2f}s")
        print(f" - 基态能量：{E[0]:.4f} meV")
        return E[0], psi[:, 0], occ_lists

    @staticmethod
    def is_only_one_element_different(list_a: list, list_b: list):
        """
        判断两个长度相同的列表看成是集合时，是否恰好只有一个元素不同
        参数:
            list_a: 第一个列表
            list_b: 第二个列表
        返回:
            bool: 符合条件返回True，否则返回False
        """
        set_a, set_b = set(list_a), set(list_b)
        # 第一步：先检查两个列表长度是否相同（前提条件）
        if len(list_a) != len(list_b):
            raise ValueError("列表长度不一致")
        # 第二步：计算[A-B, B-A]
        d_ab, d_ba = list(set_a - set_b), list(set_b - set_a)
        # 第三步：判断是否只有一个元素不同
        if len(d_ab) != 1 or len(d_ba) != 1:
            return False, [], 0
        # 如果满足条件，则返回True，差集list，以及差集元素位置决定的符号
        sgn = 1 if (list_a.index(d_ab[0]) + list_b.index(d_ba[0])) % 2 == 0 else -1
        return True, d_ab + d_ba, sgn

    def similarity_between_e_h_distribution(self, occ_lists, psi):
        """极致优化版电子-空穴分布相似度计算"""
        t_start = time.time()
        n_cc = len(occ_lists)

        if n_cc != len(psi):
            raise ValueError("occ_lists和psi长度不一致")

        # 1. 预计算所有映射和索引（极致优化版）
        n_total_orb = 2 * self.Neo + 2 * self.Nho
        neo, nho = self.Neo, self.Nho

        # 使用数组操作创建映射表
        orb_space_idx = np.zeros(n_total_orb, dtype=np.int32)
        orb_is_electron = np.zeros(n_total_orb, dtype=bool)

        # 电子↑
        orb_space_idx[:neo] = np.arange(neo)
        orb_is_electron[:neo] = True

        # 电子↓
        orb_space_idx[neo:2 * neo] = np.arange(neo)
        orb_is_electron[neo:2 * neo] = True

        # 空穴↑
        orb_space_idx[2 * neo:2 * neo + nho] = np.arange(nho)
        orb_is_electron[2 * neo:2 * neo + nho] = False

        # 空穴↓
        orb_space_idx[2 * neo + nho:2 * neo + 2 * nho] = np.arange(nho)
        orb_is_electron[2 * neo + nho:2 * neo + 2 * nho] = False

        # 2. 预计算组态信息 - 使用NumPy数组
        occ_matrix = np.array([occ for occ, _ in occ_lists], dtype=np.int32)  # (n_cc, n_orb_per_occ)

        # 3. 预计算psi的相关量
        psi_outer = np.outer(psi, psi)  # (n_cc, n_cc)
        psi_sq = np.square(psi)  # (n_cc,)

        # 4. 计算密度矩阵系数
        density_e_coef = np.zeros((neo, neo), dtype=np.float64)
        density_h_coef = np.zeros((nho, nho), dtype=np.float64)

        # 4.1 优化对角线元素计算 - 使用向量化
        # 为每个组态计算所有轨道的空间索引和类型
        for idx in range(n_cc):
            occ_array = occ_matrix[idx]

            # 批量处理所有轨道
            space_indices = orb_space_idx[occ_array]
            is_electron = orb_is_electron[occ_array]

            # 电子轨道
            e_mask = is_electron
            if np.any(e_mask):
                e_indices = space_indices[e_mask]
                np.add.at(density_e_coef, (e_indices, e_indices), psi_sq[idx])

            # 空穴轨道
            h_mask = ~e_mask
            if np.any(h_mask):
                h_indices = space_indices[h_mask]
                np.add.at(density_h_coef, (h_indices, h_indices), psi_sq[idx])

        # 4.2 优化非对角线元素计算
        # 预计算每个组态的集合表示
        # 使用frozenset避免重复计算hash
        occ_sets = [frozenset(occ) for occ in occ_matrix]

        # 使用向量化方法加速集合操作
        # 创建一个布尔掩码矩阵，标记哪些组态对可能只有一个元素不同
        # 我们先比较所有i<j的组态对
        for i in range(n_cc):
            set_i = occ_sets[i]
            occ_i = occ_matrix[i]

            # 为当前组态i预计算位置映射
            pos_map_i = {orb: pos for pos, orb in enumerate(occ_i)}

            # 与所有后续组态比较
            for j in range(i + 1, n_cc):
                set_j = occ_sets[j]

                # 快速检查：如果集合大小不同，则跳过
                if len(set_i ^ set_j) != 2:
                    continue

                # 找到不同的元素
                diff_ab = set_i - set_j
                diff_ba = set_j - set_i

                if len(diff_ab) != 1 or len(diff_ba) != 1:
                    continue

                a = next(iter(diff_ab))  # 在i中，不在j中
                b = next(iter(diff_ba))  # 在j中，不在i中

                # 检查轨道类型是否一致
                if orb_is_electron[a] != orb_is_electron[b]:
                    continue

                psi_ij = psi_outer[i, j]

                # 计算符号 - 使用预计算的位置映射
                pos_i = pos_map_i[a]
                pos_j = np.where(occ_matrix[j] == b)[0][0]
                sign = 1 if ((pos_i + pos_j) & 1) == 0 else -1  # 使用位运算判断奇偶

                # 获取空间轨道索引
                idx1, idx2 = orb_space_idx[a], orb_space_idx[b]

                # 更新密度矩阵
                value = psi_ij * sign

                if orb_is_electron[a]:  # 电子
                    density_e_coef[idx1, idx2] += value
                    density_e_coef[idx2, idx1] += value
                else:  # 空穴
                    density_h_coef[idx1, idx2] += value
                    density_h_coef[idx2, idx1] += value

        # 5. 极致优化密度分布计算
        # 使用einsum优化矩阵乘法
        # 预计算轨道乘积的张量
        orb_e_product = self.orb_e[:, :, np.newaxis] * self.orb_e[:, np.newaxis, :]  # (N_total, Neo, Neo)
        orb_h_product = self.orb_h[:, :, np.newaxis] * self.orb_h[:, np.newaxis, :]  # (N_total, Nho, Nho)

        # 使用einsum计算密度分布
        density_e = np.einsum('ij,kij->k', density_e_coef, orb_e_product, optimize='optimal')
        density_h = np.einsum('ij,kij->k', density_h_coef, orb_h_product, optimize='optimal')

        # 6. 计算粒子数和相似度
        number_e = np.sum(density_e) * self.dV
        number_h = np.sum(density_h) * self.dV
        density_e_norm = density_e / number_e
        density_h_norm = density_h / number_h

        if np.isclose(number_e, 0.0) or np.isclose(number_h, 0.0):
            cos_sim = 0.0
            dif_sim = 0.0
        else:
            # 避免数值误差导致的负数
            density_e_norm = np.maximum(density_e_norm, 1e-12)
            density_h_norm = np.maximum(density_h_norm, 1e-12)

            # 使用高效的相似度计算
            cos_sim = np.sum(np.sqrt(density_e_norm * density_h_norm)) * self.dV
            dif_sim = 1 - 0.5 * np.sum(np.abs(density_e_norm - density_h_norm)) * self.dV

        print(f" - 电子数：{number_e:.4f} | 空穴数：{number_h:.4f}")
        print(f" - 电子分布与空穴分布之间相似度")
        print(f"   余弦相似度：{cos_sim:.8f} | 差异相似度：{dif_sim:.8f}")
        print(f" - 分布计算耗时：{time.time() - t_start:.2f}s")

        return cos_sim, dif_sim, density_e_norm, density_h_norm

# ====================== 测试代码 ======================
if __name__ == "__main__":
    t_test = time.time()
    # 1. 网格参数（轻量化测试）
    #grid_size = (101, 101, 101)
    #grid_step = (0.1, 0.1, 0.1)
    grid_size = (61, 61, 61)
    grid_step = (0.1, 0.1, 0.1)
    #grid_size = (50, 50, 50)
    #grid_step = (0.12, 0.12, 0.12)
    # grid_size = (40, 40, 40)
    # grid_step = (0.15, 0.15, 0.15)
    #grid_size = (31, 31, 31)
    #grid_step = (0.2, 0.2, 0.2)
    Lx, Ly, Lz = [(s - 1) * step for s, step in zip(grid_size, grid_step)]

    # 2. 量子点掩码
    x = np.linspace(-Lx / 2, Lx / 2, grid_size[0])
    y = np.linspace(-Ly / 2, Ly / 2, grid_size[1])
    z = np.linspace(-Lz / 2, Lz / 2, grid_size[2])
    xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
    r = np.sqrt(xg ** 2 + yg ** 2 + zg ** 2)
    mask = np.where(r <= 2, 0, 1)

    # 3. 物理参数
    me_list = [0.2, 0.2]
    mh_list = [0.2, 0.2]
    Ve_list = [0.0, 1000.0]
    Vh_list = [0.0, 1000.0]
    eps_r_list = [12.0, 9.0]
    Norb_max = 8
    N_stretch, s_max, p = 20, 180, 2
    ne_up, ne_dn, nh_up, nh_dn = 1, 0, 1, 0

    # 4. 初始化+计算基态
    qds = QuantumDots(grid_size, grid_step, mask,
                              me_list, mh_list, Ve_list, Vh_list, eps_r_list,
                              Norb_max=Norb_max, N_stretch=N_stretch, s_max=s_max, p=p)
    E_gs, psi_gs, occ_lists = qds.get_ground_state(ne_up, ne_dn, nh_up, nh_dn, use_sparse=True)
    qds.similarity_between_e_h_distribution(occ_lists, psi_gs)

    # 输出关键结果
    print("=" * 50)
    print(f"电子能级（前{len(qds.Eeo)}个）：{np.round(qds.Eeo, 10)}")
    print(f"空穴能级（前{len(qds.Eho)}个）：{np.round(qds.Eho, 10)}")
    print(f"最终基态能量：{E_gs:.10f} meV, {N_stretch, s_max, p}")

    print(f"测试总耗时：{time.time() - t_test:.2f}s")

'''

'''
