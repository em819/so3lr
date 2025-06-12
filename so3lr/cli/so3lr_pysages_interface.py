"""
SO3LR interface to PySAGES

This module provides the interface to PySAGES to perform MDs with enhanced sampling
and collective variables
"""

import pysages
import pysages.backends as pb
import pysages.utils as pu

def create_pysages_interface_fns(lr, state, box, step_md_fn, md_dt, nbrs, nbrs_lr=None):
    if lr:
        def init_fn_pysages():
            return pb.JaxMDContextState(state,dict(nbrs=nbrs, nbrs_lr=nbrs_lr, box=box))
    else:
        def init_fn_pysages():
            return pb.JaxMDContextState(state, dict(nbrs=nbrs, box=box))

    if lr:
        def step_fn_pysages(context_state):
            state = context_state.state
            nbrs = context_state.extras["nbrs"]
            box = context_state.extras["box"]
            i = 0 #Set md_step variable to 0, since it's not used inside anyway

            nbrs_lr = context_state.extras["nbrs_lr"]
            state, nbrs, nbrs_lr, box= step_md_fn(i, (state, nbrs, nbrs_lr, box))
            #print(f'Forces : {state.force}')
            return pb.JaxMDContextState(state,dict(nbrs=nbrs, nbrs_lr=nbrs_lr, box=box))
    else:
        def step_fn_pysages(context_state):
            state = context_state.state
            nbrs = context_state.extras["nbrs"]
            box = context_state.extras["box"]
            i = 0 #Set md_step variable to 0, since it's not used inside anyway

            state = step_md_fn(i, state, nbrs, box)
            return pb.JaxMDContextState(state,dict(nbrs=nbrs, box=box))

    def generate_context_pysages():
        return pb.JaxMDContext(init_fn_pysages, step_fn_pysages, box, md_dt)

    return generate_context_pysages

def update_so3lr_after_pysages(raw_result, lr, init_fn, rng_key, md_T, neighbor_fn, neighbor_fn_lr=None):
    def get_new_momenta(snapshot):
        V, M = snapshot.vel_mass
        return (V * M).flatten()

    def get_masses(snapshot):
        _, M = snapshot.vel_mass
        return M

    def get_velocities(snapshot):
        V, _ = snapshot.vel_mass
        return V

    print(f'Number of snapshots {len(raw_result.snapshots)}')

    final_snapshot = raw_result.snapshots[-1]

    new_box = final_snapshot.box.H

    nbrs = neighbor_fn.allocate(final_snapshot.positions, box=new_box)
    nbrs_lr = neighbor_fn_lr.allocate(final_snapshot.positions, box=new_box) if lr else None

    #Create so3lr-compatible state after pysages returns via init_fn
    if lr:
        new_state = init_fn(
            rng_key,
            R=final_snapshot.positions,
            box=new_box,
            neighbor=nbrs.idx,
            neighbor_lr=nbrs_lr.idx,
            kT=md_T,
            mass=get_masses(final_snapshot),
            velocities=get_velocities(final_snapshot)
        )
    else:
        new_state = init_fn(
            rng_key,
            R=final_snapshot.positions,
            box=new_box,
            neighbor=nbrs.idx,
            kT=md_T,
            mass=get_masses(final_snapshot),
            velocities=get_velocities(final_snapshot)
        )

    return new_state, nbrs, nbrs_lr, new_box

def parse_pysages_input(input_path):
    """ 
    Read settings for choice of CVs, restraints, grid, sampling method
    """

    with open(input_path, 'r') as in_file:
        settings_dict = {}
        for line in in_file:
            line = line.lower().strip()

            if line.startswith('#'):
                continue
            elif line.startswith('method '):
                settings_dict['method'] = line.split()[1]
            elif line.startswith('method_args'):
                line_vals = line.split()
                if method_args not in settings_dict:
                    settings_dict['method_args'] = {}
                
                settings_dict['method_args'][linevals[1]] = linevals[2]

            elif line.startswith('cv'):
                line_vals = line.split()
                if 'cv' not in settings_dict:
                    settings_dict['cv'] = []

                if any(line_vals[1] == x for x in ['distance']): 
                #Expecting: distance [0,1,2,3,...] [10,11,12,13,...]
                    cv_dict = {'type': 'distance', 'grp1': line_vals[2], 'grp2' : line_vals[3]}
                    settings_dict['cv'].append(cv_dict)
            
    return settings_dict


def create_pysages_runner(method, generate_context_fn, md_steps):
    """ 
    Return .run function to perform selected enhanced sampling method
    """

    pass

def save_pysages_state(pysages_result, path_to_save):
    """ 
    Create restart file for pysages. For a clean restart, one only needs to save the
    Result object after a pysages run
    """
    
    pu.save(pysages_result, path_to_save)
