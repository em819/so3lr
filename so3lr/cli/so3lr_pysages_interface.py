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
import jax_md.units
from jax import jit
import pysages.colvars
import pysages.methods
import json
import numpy as np
import jax.numpy as jnp
import yaml

def create_pysages_interface_fns(lr, state, box, step_md_fn, md_dt, nbrs, nbrs_lr=None):
    #print(f'Box : {box} (shape : {box.shape}, type={type(box)})')
    if np.all(box == 0) or box is None:
        box = jnp.array([0.0, 0.0, 0.0])
        #state.box = box
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
        settings_dict = yaml.load(in_file, Loader=yaml.SafeLoader)
        #settings_dict = {}
        #for line in in_file:
        #    line = line.lower().strip()

        #    if line.startswith('#'):
        #        continue
        #    elif line.startswith('method '):
        #        settings_dict['method'] = line.split()[1]
        #    elif line.startswith('method_args'):
        #        line_vals = line.split()
        #        if 'method_args' not in settings_dict:
        #            settings_dict['method_args'] = {}
        #        
        #        settings_dict['method_args'][line_vals[1]] = line_vals[2]

        #    elif line.startswith('cv'):
        #        line_vals = line.split()
        #        if 'cv' not in settings_dict:
        #            settings_dict['cv'] = []

        #        if any(line_vals[1] == x for x in ['distance']): 
        #        #Expecting: distance [0,1,2,3,...] [10,11,12,13,...] 0.0 50.0 64
        #            cv_dict = {
        #                    'type': 'distance', 
        #                    'grp1': json.loads(line_vals[2]), 
        #                    'grp2' : json.loads(line_vals[3]),
        #                    'grid_min' : float(line_vals[4]),
        #                    'grid_max' : float(line_vals[5]),
        #                    'grid_stride' : int(line_vals[6]),
        #                    'periodic' : False
        #                    }
        #        elif line_vals[1] == 'angle':
        #        #Expecting: angle [2,3,4]
        #            cv_dict = {
        #                    'type': 'angle', 
        #                    'indices': json.loads(line_vals[2]),
        #                    'grid_min' : line_vals[3],
        #                    'grid_max' : line_vals[4],
        #                    'grid_stride' : line_vals[5],
        #                    'periodic' : True
        #                    }
        #        elif line_vals[1] == 'dihedral':
        #        #Expecting: dihedral [4,5,6,7]
        #            cv_dict = {
        #                    'type': 'dihedral', 
        #                    'indices': json.loads(line_vals[2]),
        #                    'grid_min' : line_vals[3],
        #                    'grid_max' : line_vals[4],
        #                    'grid_stride' : line_vals[5],
        #                    'periodic' : True
        #                    }
        #        elif line_vals[1] == 'gyrrad':
        #        #Expecting: gyrrad [2,4,6,7,8,9,...]
        #            cv_dict = {
        #                    'type': 'gyrrad', 
        #                    'indices': json.loads(line_vals[2]),
        #                    'grid_min' : line_vals[3],
        #                    'grid_max' : line_vals[4],
        #                    'grid_stride' : line_vals[5],
        #                    'periodic' : False
        #                    }
        #        elif line_vals[1] == 'princmom':
        #        #Expecting: princmom [2,3,4,5,6,...] 1
        #            cv_dict = {
        #                    'type': 'princmom', 
        #                    'indices': json.loads(line_vals[2]), 
        #                    'axis' : line_vals[3],
        #                    'grid_min' : line_vals[4],
        #                    'grid_max' : line_vals[5],
        #                    'grid_stride' : line_vals[6],
        #                    'periodic' : False
        #                    }
        #        elif line_vals[1] == 'asphericity':
        #        #Expecting: asphericity [1,2,3,4,...]
        #            cv_dict = {
        #                    'type': 'asphericity', 
        #                    'indices': json.loads(line_vals[2]),
        #                    'grid_min' : line_vals[3],
        #                    'grid_max' : line_vals[4],
        #                    'grid_stride' : line_vals[5],
        #                    'periodic' : False
        #                    }
        #        elif line_vals[1] == 'acylindricity':
        #        #Expecting: acylindricity [1,2,3,4,...]
        #            cv_dict = {
        #                    'type': 'acylindricity', 
        #                    'indices': json.loads(line_vals[2]),
        #                    'grid_min' : line_vals[3],
        #                    'grid_max' : line_vals[4],
        #                    'grid_stride' : line_vals[5],
        #                    'periodic' : False
        #                    }
        #        elif line_vals[1] == 'shapeanisotropy':
        #            #Expecting: shapeanisotropy [1,2,3,4,...]
        #            cv_dict = {
        #                    'type': 'shapeanisotropy', 
        #                    'indices': json.loads(line_vals[2]),
        #                    'grid_min' : line_vals[3],
        #                    'grid_max' : line_vals[4],
        #                    'grid_stride' : line_vals[5],
        #                    'periodic' : False
        #                    }
        #        settings_dict['cv'].append(cv_dict)



    print(f'Parsed settings for PySAGES : {settings_dict}')       
    return settings_dict

def process_cv(cv_dict, nbrs=None, species=None, box=None):
    name = cv_dict['type']
    arguments = cv_dict

    name_lower = name.lower()
    if name_lower == 'distance':
        #print(f'Grp 1: {tuple(arguments['grp1'])}, Grp 2: {tuple(arguments['grp2'])}')
        return pysages.colvars.Distance(indices=[tuple(arguments['grp1']), tuple(arguments['grp2'])])
    elif name_lower == 'angle':
        return pysages.colvars.Angle(indices=arguments['indices'])
    elif name_lower == 'dihedral':
        return pysages.colvars.DihedralAngle(indices=arguments['indices'])
    elif name_lower == 'gyrrad':
        return pysages.colvars.RadiusOfGyration(indices=arguments['indices'])
    elif name_lower == 'princmom':
        return pysages.colvars.PrincipalMoment(indices=arguments['indices'], axis=arguments['axis'])
    elif name_lower == 'asphericity':
        return pysages.colvars.Asphericity(indices=arguments['indices'])
    elif name_lower == 'acylindricity':
        return pysages.colvars.Acylindricity(indices=arguments['indices'])
    elif name_lower == 'shapeanisotropy':
        return pysages.colvars.ShapeAnisotropy(indices=arguments['indices'])
    elif name_lower == 'coordinationnumber':
        n_atoms = nbrs.reference_position.shape[0]
        return pysages.colvars.coordinates.CoordinationNumber(
                indices=[tuple(np.arange(0,n_atoms).tolist()), tuple(arguments['indices'])], 
                nbrs=nbrs, 
                species=species, 
                box=box,
                species_nn=arguments['species'],
                cn_exponents=arguments['cn_exps'] if 'cn_exps' in arguments else None
                )
        #return pysages.colvars.coordinates.CoordinationNumber(indices=[tuple(indices_min), tuple(arguments['indices'])], nbrs=nbrs_min, species=species, box=box, max_neighbors=max_neighbors)

def process_grid(cv_settings):
    grid_mins = [cv['grid_min'] for cv in cv_settings]
    grid_maxs = [cv['grid_max'] for cv in cv_settings]
    grid_strides = [cv['grid_stride'] for cv in cv_settings]
    periodic = [cv['periodic'] for cv in cv_settings]
    #print(f'Grid shape : {tuple(grid_strides)}')
    return pysages.Grid(lower=tuple(grid_mins), upper=tuple(grid_maxs), shape=tuple(grid_strides), periodic=all(periodic))



def get_pysages_method(settings_dict, nbrs=None, species=None, box=None):
    """Process the parsed settings and return the enhanced sampling method object"""
    #First process the CVs
    cvs = [process_cv(cv, nbrs=nbrs, species=species, box=box) for cv in settings_dict['cv']]
    #Then collect the grid information
    grid = process_grid(settings_dict['cv'])

    units = jax_md.units.metal_unit_system()
    #print(units)

    if 'restraints' in settings_dict:
        restraints_dict = settings_dict['restraints']
        restraints = pysages.CVRestraints(lower=restraints_dict['lower'], upper=restraints_dict['upper'], kl=0, ku=0.1)
    else:
        restraints=None

    #Assemble method
    if settings_dict['method'].lower() == 'abf':
        return pysages.methods.ABF(cvs, grid, restraints=restraints)
    elif settings_dict['method'].lower() == 'metad':
        return pysages.methods.Metadynamics(
                cvs, 
                settings_dict['method_args']['height'], 
                settings_dict['method_args']['sigma'], 
                settings_dict['method_args']['stride'],
                settings_dict['method_args']['ngauss'], 
                deltaT=settings_dict['method_args']['deltaT'], 
                kB=units['temperature'], 
                grid=grid,
                restraints=restraints)
    elif settings_dict['method'].lower() == 'unbiased':
        return pysages.methods.Unbiased(cvs)
    else:
        #TODO implement the other methods
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

