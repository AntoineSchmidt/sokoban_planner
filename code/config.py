import platform


class Config:
    silent = True

    # search
    early_stop = True # astar: quits in expansion step, mcts_improved: quits on rollout found solution

    # reward
    custom = True
    discount = 0.98
    discount_tree = 1.0

    # fast forward
    timeout = 5
    wsl = (platform.system() == 'Windows')

    # best found networks
    net_1 = ([(26, 3), (23, 6), (20, 7), (29, 8, 2), (30, 5, 2), (31, 7), (10, 5)], [])
    net_1_small = ([(9, 3), (11, 7), (14, 8), (9, 4), (15, 8), (12, 9), (13, 8)], [42])

    net_4 = ([(29, 6), (24, 4), (26, 8), (16, 5), (22, 4, 2), (23, 7)], [14])
    net_4_small = ([(16, 5), (15, 7), (6, 6), (16, 4), (7, 4), (6, 8), (6, 9), (16, 5)], [83])