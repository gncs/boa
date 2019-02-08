import time

import numpy as np
from aquisition.smsego import SMSego
from models.gp.gplib import GPlib
from models.random.randommodel import RandomModel
from optimizer_helper import InvalidParameters, shift, find_frontier, get_hypervolume


def optimize(f,
             candidates,
             model_params,
             aquisition_params,
             num_init,
             num_max,
             reference_point,
             batch_size=1,
             output_dir=None,
             plots=False):

    print('Initializing values')

    # Points of the initial evaluations
    init_index = []
    init_values = []
    np.random.seed(69420)
    for i in range(num_init):
        index = np.random.randint(0, candidates.shape[0])
        try:
            value = f(candidates[index, :])
            init_index.append(index)
            init_values.append(value)
            print(value)
        except InvalidParameters:
            i -= 1	
    init_points = candidates[init_index, :]
    candidates = np.delete(candidates, init_index, 0)

    init_values = np.reshape(np.array(init_values), (len(init_values), -1))
    init_values_mean = np.mean(init_values, axis=0)
    init_values_std = np.std(init_values, axis=0) + 1e-10
    print(zip(init_points, init_values))

    # Model
    print('Initializing the Predictive Model')

    if model_params.name == 'gp':
        model = GPlib(init_points, init_values, model_params)
    elif model_params.name == 'rng':
        model = RandomModel(init_points, init_values, model_params)
    else:
        raise Exception('Ill-specified model name')

    # Aquisition
    print('Initializing the Acquisition Function')

    if aquisition_params.name == 'smsego':
        aquisition_function = SMSego(aquisition_params, reference_point, init_values_mean, init_values_std)
    else:
        raise Exception('Ill-specified aquisition function')

    frontier = find_frontier(init_values)
    hv_over_iterations = [aquisition_function.get_hypervolume(frontier, reference_point)]

    # Iteration
    print('Iterating')
    current_points = init_points.copy()
    current_values = init_values.copy()
    iter = 0
    while iter < num_max:
        iter += 1
        print('Iteration {0}'.format(iter))

	eval_points = []
	for b in range(batch_size):
            print('Selecting highest aq value')
            aquisition_values = aquisition_function.getAquisitionBatch(candidates, model, frontier)
            max_aquisition_index = np.argmax(aquisition_values)
            new_point = candidates[max_aquisition_index]
	    eval_points.append(new_point)
            print('Adding new pseudo point')
            model.addPseudoPoint(new_point)
            candidates = np.delete(candidates, max_aquisition_index, 0)

	model.removePseudoPoints()

	# Parallelize this
        for ep in eval_points:
	    try:
                print('Evaluating point')
                new_point_value = np.reshape(np.array(f(ep)), (1, -1))
                current_values = np.vstack((new_point_value, current_values))
                print(new_point_value)
                current_points = np.vstack((ep, current_points))
                print('Updating model')
                model.addPoint(ep, new_point_value)
                frontier = find_frontier(np.vstack((new_point_value, frontier)))
            except InvalidParameters:
                iter -= 1
                print('Invalid Parameters')

        hv_over_iterations.append(aquisition_function.get_hypervolume(frontier, reference_point))
        print('   Hypervolume improved from {0} to {1}'.format(hv_over_iterations[-2], hv_over_iterations[-1]))
   
    print('The final Hypervolume is {0}'.format(hv_over_iterations[-1]))
    return frontier, np.array(hv_over_iterations), np.array(current_values)

