import sys
import pickle
import numpy as np
import tensorflow as tf

import logging
logging.basicConfig(level=logging.ERROR)

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from utils import *
from manage import *
from network import *


class MyWorker(Worker):
    def parse(config):
        network = ([], [])
        for i in range(config['num_c']):
            layer = (config['c_f-{}'.format(i)], config['c_s-{}'.format(i)])
            layer_p = config['c_p-{}'.format(i)]
            if layer_p > 1: layer += (layer_p,)
            network[0].append(layer)

        for i in range(config['num_l']):
            network[1].append(config['l-{}'.format(i)])

        return network


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # load data
        with open('data/imitation_data.pkl', 'rb') as f:
            self.data_val = pickle.load(f)
        file = 'data/imitation_search_{}.npy'
        self.data_state = np.load(file.format('state'))
        self.data_action = np.load(file.format('action'))
        self.data_length = np.load(file.format('length'))

        self.setting = 0
        self.networks = []


    def compute(self, config, budget, **kwargs):
        self.setting += 1
        try:
            network = MyWorker.parse(config)

            startSession()
            model, _ = create(model=network, shape_in=(13, 13, 4), shape_out=1 if optimize == 'length' else 4,
                              activation_out='linear' if optimize == 'length' else 'softmax',
                              loss='costum' if optimize == 'length' else None)
            history = model.fit(self.data_state, self.data_length if optimize == 'length' else self.data_action,
                                epochs=int(budget), batch_size=256,
                                validation_data=(self.data_val['state_val'], self.data_val[optimize + '_val']),
                                class_weight=weights(self.data_length) if optimize == 'length' else None).history
            del model
            finishSession()

            loss = min(history['loss'])
            val_loss = min(history['val_loss']) # ignore overfitting
            val_acc = None if optimize == 'length' else max(history['val_categorical_accuracy'])
            self.networks.append([loss, val_loss, val_acc, network])
            print(self.setting, network, loss, val_loss, val_acc)
            return ({'loss': val_loss, 'info': {}})
        except Exception as e:
            print(str(e))
            print(history.keys())


    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()

        # conv layer
        num_c =  CSH.UniformIntegerHyperparameter('num_c', lower=4, upper=10)
        config_space.add_hyperparameters([num_c])
        for i in range(num_c.upper):
            c_filter = CSH.UniformIntegerHyperparameter('c_f-{}'.format(i), lower=1, upper=32)
            c_size = CSH.UniformIntegerHyperparameter('c_s-{}'.format(i), lower=3, upper=9)
            c_pooling = CSH.UniformIntegerHyperparameter('c_p-{}'.format(i), lower=1, upper=2)
            config_space.add_hyperparameters([c_filter, c_size, c_pooling])
            if i > num_c.lower:
                config_space.add_condition(CS.GreaterThanCondition(c_filter, num_c, i))
                config_space.add_condition(CS.GreaterThanCondition(c_size, num_c, i))
                config_space.add_condition(CS.GreaterThanCondition(c_pooling, num_c, i))

        # fully connected layer
        num_l =  CSH.UniformIntegerHyperparameter('num_l', lower=0, upper=1)
        config_space.add_hyperparameters([num_l])
        for i in range(num_l.upper):
            layer = CSH.UniformIntegerHyperparameter('l-{}'.format(i), lower=10, upper=100)
            config_space.add_hyperparameters([layer])
            config_space.add_condition(CS.GreaterThanCondition(layer, num_l, i))

        return config_space


if __name__ == "__main__":
    try: # read in optimization choice
        optimize = str(sys.argv[1])
        assert(optimize == 'action' or optimize == 'length')
    except:
        print('Choose action (default) or length to optimize')
        optimize = 'action'

    ID = 'search_{}'.format(optimize)
    BUDGET = 10
    ITERATIONS = 100

    NS = hpns.NameServer(run_id=ID, host='127.0.0.1', port=None)
    NS.start()
    WK = MyWorker(nameserver='127.0.0.1', run_id=ID)
    WK.run(background=True)

    BS = BOHB(configspace=WK.get_configspace(), run_id=ID, nameserver='127.0.0.1', min_budget=BUDGET, max_budget=BUDGET)
    BSP = BS.run(n_iterations=ITERATIONS)

    BS.shutdown(shutdown_workers=True)
    NS.shutdown()

    # save and show results
    with open('data/search_{}.pkl'.format(optimize), 'wb') as fp:
        pickle.dump(WK.networks, fp)

    WK.networks.sort(key = lambda entry: entry[1])
    for i in range(min(5, len(WK.networks))):
        print(WK.networks[i])