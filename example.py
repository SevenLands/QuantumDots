import numpy as np
from numpy.random import shuffle, choice
import pickle
import time
from qd import QuantumDots
from matplotlib import pyplot as plt

def generate_grid(box_size, ini_grid_step=0.1):
    igs = ini_grid_step
    grid_size = int(box_size[0] / igs) + 1, int(box_size[0] / igs) + 1, int(box_size[0] / igs) + 1
    grid_step = box_size[0] / (grid_size[0] - 1), box_size[0] / (grid_size[1] - 1), box_size[0] / (grid_size[2] - 1)
    return grid_size, grid_step

def generate_mask(box_size, grid_size, geo, n_material, qd_type):
    # 2. 量子点掩码
    x = np.linspace(- box_size[0] / 2, box_size[0] / 2, grid_size[0])
    y = np.linspace(- box_size[1] / 2, box_size[1] / 2, grid_size[1])
    z = np.linspace(- box_size[2] / 2, box_size[2] / 2, grid_size[2])
    xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
    mask = np.zeros_like(xg)  # 先假设是第一个材料，都是0，后面不同一次加一次
    if qd_type == 'sphere':
        rg = np.sqrt(xg ** 2 + yg ** 2 + zg ** 2)
        for i in range(n_material):
            mask += (rg > geo[i][0])
    elif qd_type == 'cylinder':
        rg = np.sqrt(xg ** 2 + yg ** 2)
        for i in range(n_material):
            mask += (rg > geo[i][0]) | (np.abs(zg) > geo[i][1])
    elif qd_type == 'rectangle':
        for i in range(n_material):
            mask += (np.abs(xg) > geo[i][0]) | (np.abs(yg) > geo[i][1]) | (np.abs(zg) > geo[i][2])
    elif qd_type == 'ellipsoid':
        for i in range(n_material):
            judge = (xg / geo[i][0]) ** 2 + (yg / geo[i][1]) ** 2 + (zg / geo[i][2]) ** 2
            mask += (judge > 1)
    return mask

def main(qd_type, geo,
         me_list, mh_list, Ve_list, Vh_list, eps_r_list,
         Norb_max=None, box_size=None, ini_grid_step=None,
         N_stretch=20, s_max=180, p=2):
    # 参数: qd_type: quantum dot type, 'sphere', 'cylinder', 'rectangle'
    # 参数: n_shell: number of shell layers
    # 参数: box_size: box size, [Lx, Ly, Lz]

    # 计算网格参数
    Norb_max = 8 if Norb_max is None else Norb_max
    box_size = [10, 10, 10] if box_size is None else box_size
    ini_grid_step = 0.1 if ini_grid_step is None else ini_grid_step
    grid_size, grid_step = generate_grid(box_size, ini_grid_step=ini_grid_step)  # 网格大小和步长

    n_shell = len(me_list) - 1

    if n_shell < 1:
        raise ValueError('未设置壳层材料')

    print('n_shell:', n_shell, flush=True)
    print(' - me_list:', me_list, flush=True)
    print(' - mh_list:', mh_list, flush=True)
    print(' - eps_r_list:', eps_r_list, flush=True)
    print(' - Ve_list:', Ve_list, flush=True)
    print(' - Vh_list:', Vh_list, flush=True)
    print(' - geo:', geo, flush=True)
    print(' - grid_size:', grid_size)
    print(' - grid_step:', grid_step)
    # 材料mask
    mask = generate_mask(box_size, grid_size, geo, n_shell, qd_type)
    print(' - mask:', mask.shape, flush=True)
    '''
    dim2 = mask.shape[0]//2
    mask_yz = mask[dim2, :, :]
    plt.imshow(mask_yz)
    plt.title('mask_yz')
    plt.show()
    mask_xz = mask[:, dim2, :]
    plt.imshow(mask_xz)
    plt.title('mask_xz')
    plt.show()
    mask_xy = mask[:, :, dim2]
    plt.imshow(mask_xy)
    plt.title('mask_xy')
    plt.show()
    '''
    #pickle.dump(mask, open(f'mask_{qd_type}.pkl', 'wb'))
    #raise Exception('for mask, stop.')

    qds = QuantumDots(grid_size, grid_step, mask,
                              me_list, mh_list, Ve_list, Vh_list, eps_r_list,
                              Norb_max=Norb_max, N_stretch=N_stretch, s_max=s_max, p=p)

    print(f"\n 计算 1e1h 态 ... ")
    E_1e1h, psi_1e1h, occ_1e1h = qds.get_ground_state(1, 0, 1, 0, use_sparse=False)
    Eb_1e1h = qds.Eeo[0] + qds.Eho[0] - E_1e1h
    cos_sim_1e1h, dif_sim_1e1h = qds.similarity_between_e_h_distribution(occ_1e1h, psi_1e1h)

    print("=" * 50, flush=True)
    print(f"电子能级（前{len(qds.Eeo)}个）：{np.round(qds.Eeo, 2)}", flush=True)
    print(f"空穴能级（前{len(qds.Eho)}个）：{np.round(qds.Eho, 2)}", flush=True)
    print(f"最终束缚能能量：\n"
          f"1e1h：{Eb_1e1h:.4f} meV\n")

    return (geo,
            np.array([me_list, mh_list, Ve_list, Vh_list, eps_r_list]).tolist(),
            np.array([qds.Eeo[:4], qds.Eho[:4]]).tolist(),
            np.array([Eb_1e1h, cos_sim_1e1h, dif_sim_1e1h]).tolist())

if __name__ == '__main__':
    qd_type = 'rectangle'  # cylinder  sphere  rectangle  ellipsoid

    #side_length = np.arange(1, 4.1, 0.5)
    side_length = [2]
    out = []
    for sd in side_length:
        geo = [[sd, sd, sd]]  # rectangle
        #geo = [[sd]]  # sphere

        me_list, mh_list = [0.5, 0.5], [0.5, 0.5]
        Ve_list, Vh_list = [0.0, 1e8], [0.0, 1e8]
        eps_r_list = [10, 10]

        Norb_max = 2
        box_size = [10, 10, 10]
        ini_grid_step = 0.1

        N_stretch, s_max, p = 20, 50, 2

        # 运行
        t0 = time.time()
        result = main(qd_type, geo,
                      me_list, mh_list, Ve_list, Vh_list, eps_r_list,
                      Norb_max=Norb_max, box_size=box_size, ini_grid_step=ini_grid_step,
                      N_stretch=N_stretch, s_max=s_max, p=p)
        out.append(result)
        with open(qd_type + '.pkl', "wb") as f:
            pickle.dump(out, f)

        print(f"总用时：{time.time() - t0:.2f} s", flush=True)
    with open(qd_type + '.pkl', "wb") as f:
        pickle.dump(out, f)
