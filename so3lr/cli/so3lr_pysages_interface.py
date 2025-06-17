"""
SO3LR interface to PySAGES

This module provides the interface to PySAGES to perform MDs with enhanced sampling
and collective variables
"""

import pysages
import pysages.backends as pb
import pysages.utils as pu
import jax_md.simulate
import jax_md.quantity
from jax import jit

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
            return pb.JaxMDContextState(state,dict(nbrs=nbrs, nbrs_lr=nbrs_lr, box=box))
    else:
        def step_fn_pysages(context_state):
            state = context_state.state
            nbrs = context_state.extras["nbrs"]
            box = context_state.extras["box"]
            i = 0 #Set md_step variable to 0, since it's not used inside anyway

            state, nbrs, box = step_md_fn(i, state, nbrs, box)
            return pb.JaxMDContextState(state,dict(nbrs=nbrs, box=box))

    def generate_context_pysages():
        return pb.JaxMDContext(init_fn_pysages, step_fn_pysages, box, md_dt)

    return generate_context_pysages

@jit
def re_init_so3lr_state(chain_pos, chain_mom, chain_mass, chain_tau, chain_ekin, chain_dof, pos, velocities, forces, mass):
    new_state = jax_md.simulate.NVTNoseHooverState(
            position=pos,
            momentum=None,
            force=forces,
            mass=mass,
            chain=None)

    new_state = jax_md.simulate.canonicalize_mass(new_state)

    new_state = new_state.set(momentum=new_state.mass * velocities)

    #KE = jax_md.simulate.kinetic_energy(new_state)
    dof = jax_md.quantity.count_dof(pos)
    new_thermostat = jax_md.simulate.NoseHooverChain(
                position=chain_pos,
                momentum=chain_mom,
                mass=chain_mass,
                tau=chain_tau,
                kinetic_energy=chain_ekin,
                degrees_of_freedom=dof
            )


    new_state = new_state.set(chain=new_thermostat)
    return new_state


def update_so3lr_after_pysages(raw_result, lr, init_fn, rng_key, md_T, nbrs, nbrs_lr=None):
    def get_new_momenta(snapshot):
        V, M = snapshot.vel_mass
        return (V * M).flatten()

    def get_masses(snapshot):
        _, M = snapshot.vel_mass
        return M

    def get_velocities(snapshot):
        V, _ = snapshot.vel_mass
        return V


    final_snapshot = raw_result.snapshots[-1]

    new_box = final_snapshot.box.H

    nbrs = nbrs.update(final_snapshot.positions, neighbor=nbrs.idx, box=new_box)
    nbrs_lr = nbrs_lr.update(final_snapshot.positions, neighbor=nbrs_lr.idx, box=new_box) if lr else None

    #Create so3lr-compatible state after pysages returns via init_fn
    new_state = re_init_so3lr_state(
            final_snapshot.chain_data["position"], 
            final_snapshot.chain_data["momentum"],
            final_snapshot.chain_data["mass"], 
            final_snapshot.chain_data["tau"], 
            final_snapshot.chain_data["kinetic_energy"], 
            final_snapshot.chain_data["degrees_of_freedom"], 
            final_snapshot.positions, 
            get_velocities(final_snapshot), 
            final_snapshot.forces, 
            get_masses(final_snapshot)
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
                if 'method_args' not in settings_dict:
                    settings_dict['method_args'] = {}
                
                settings_dict['method_args'][line_vals[1]] = line_vals[2]

            elif line.startswith('cv'):
                line_vals = line.split()
                if 'cv' not in settings_dict:
                    settings_dict['cv'] = []

                if any(line_vals[1] == x for x in ['distance']): 
                #Expecting: distance [0,1,2,3,...] [10,11,12,13,...]
                    cv_dict = {'type': 'distance', 'grp1': line_vals[2], 'grp2' : line_vals[3]}
                elif line_vals[1] == 'angle':
                #Expecting: angle [2,3,4]
                    cv_dict = {'type': 'angle', 'indices': line_vals[2]}
                elif line_vals[1] == 'dihedral':
                #Expecting: dihedral [4,5,6,7]
                    cv_dict = {'type': 'dihedral', 'indices': line_vals[2]}
                elif line_vals[1] == 'gyrrad':
                #Expecting: gyrrad [2,4,6,7,8,9,...]
                    cv_dict = {'type': 'gyrrad', 'indices': line_vals[2]}
                elif line_vals[1] == 'princmom':
                #Expecting: princmom [2,3,4,5,6,...] 1
                    cv_dict = {'type': 'princmom', 'indices': line_vals[2], 'axis' : line_vals[3]}
                elif line_vals[1] == 'asphericity':
                #Expecting: asphericity [1,2,3,4,...]
                    cv_dict = {'type': 'asphericity', 'indices': line_vals[2]}
                elif line_vals[1] == 'acylindricity':
                #Expecting: acylindricity [1,2,3,4,...]
                    cv_dict = {'type': 'acylindricity', 'indices': line_vals[2]}
                elif line_vals[1] == 'shapeanisotropy':
                    #Expecting: shapeanisotropy [1,2,3,4,...]
                    cv_dict = {'type': 'shapeanisotropy', 'indices': line_vals[2]}
                settings_dict['cv'].append(cv_dict)



    print(f'Parsed settings for PySAGES : {settings_dict}')       
    return settings_dict

def get_pysages_method(settings_dict):
    """Process the parsed settings and return the enhanced sampling method object"""
    if settings_dict['method'] == 'ABF':
        pass


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
    
    #pu.save(pysages_result, path_to_save)
    pysages.serialization.save(pysages_result, path_to_save)

def load_pysages_state(path_to_load):
    return pysages.serialization.load(path_to_load)

