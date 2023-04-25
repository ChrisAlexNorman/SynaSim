import numpy as np
from itertools import combinations
import json
import jsbeautifier
from scipy.integrate import odeint

class MarkovModel:

    def __init__(self, rate_constants=np.array([0]), stim_dependence=np.array([0]), initial_condition=np.array([1]), metadata={}):
        self.rate_constants = rate_constants
        self.stim_dependence = stim_dependence
        self.initial_condition = initial_condition
        self.n_states = self.rate_constants.shape[0]
        self.metadata = self._validate_metadata(metadata)
        self.simulations = []

    def __str__(self):
        return self.metadata['name'] + f": {self.n_states}-state Markov chain"

    def __len__(self):
        return self.n_states

    def _validate_metadata(self, metadata):
        if 'name' not in metadata:
            metadata['name'] = 'Unnamed'
        if 'time_units' not in metadata:
            metadata['time_units'] = 'ms (assumed)'
        if 'stim_units' not in metadata:
            metadata['stim_units'] = 'uM (assumed)'
        if 'state_names' not in metadata:
            metadata['state_names'] = ['S'+str(n) for n in range(0,self.n_states)]
        return metadata
    
    def _clear_simulations(self):
        simulations = self.simulations
        self.simulations = []
        return simulations

    def _update_q_matrix(self, stimulus_t, rates_q=None):
        if rates_q is None:
            rates_q = np.copy(self.rate_constants)
        rates_q[self.stim_dependence==1] = self.rate_constants[self.stim_dependence==1] * stimulus_t
        rates_q[np.diag_indices(self.n_states)] = -np.sum(rates_q, axis=1)
        return rates_q

    def _ode_system(self, p, t, time, stimulus):
        """Return dp/dt at time t for time-dependent transition rates"""
        rates_q = self._update_q_matrix(np.interp(t, time, stimulus))
        return np.dot(p, rates_q)

    def get_sim_type(self, simulation):
        sim_type = 'UNKNOWN'
        sim_desc = f"Stimulus input not recognised."
        if len(simulation[0]) == 1 and len(simulation[1]) == 1:
            sim_type = 'constant'
            sim_desc = f"Constant stimulation of {simulation[1][0]} {self.metadata['stim_units']} for {simulation[0][0]} {self.metadata['time_units']}"
        elif len(simulation[0]) > 1:
            if len(np.shape(simulation[1])) == 1:
                sim_type = '1d_timeseries'
                sim_desc = f"1D trace lasting {simulation[0][-1]} {self.metadata['time_units']}"
            elif len(np.shape(simulation[1])) == 3:
                sim_type = '3d_timeseries'
                sim_desc = f"3D timeseries lasting {simulation[0][-1]} {self.metadata['time_units']}"

        return sim_type, sim_desc

    def run_simulations(self, simulations=[]):
        """Run each simulation in simulations."""

        for simulation in simulations:

            sim_type, sim_desc = self.get_sim_type(simulation)

            assert sim_type != 'UNKNOWN', sim_desc

            time = simulation[0]
            stimulus = simulation[1]

            if sim_type == 'constant':
                if len(simulation) > 2:
                    n_points = simulation[2]
                else:
                    n_points = int(1e3)
                states = self.simulate_ode_constant(time[0], stimulus[0], n_points)
            
            elif sim_type == '1d_timeseries':
                states = self.simulate_ode_trace(time, stimulus)
            
            elif sim_type == '3d_timeseries':
                states = np.zeros((np.shape(stimulus)[0], self.n_states, np.shape(stimulus)[1], np.shape(stimulus)[2]))
                for row_idx in range(0,np.shape(stimulus[0])[0]):
                    for col_idx in range(0,np.shape(stimulus[0])[1]):
                        states[:, :, row_idx, col_idx] = self.simulate_ode_trace(time, stimulus[:, row_idx, col_idx])

            self.simulations.append({\
            'type': sim_type,\
            'description': sim_desc,\
            'simulation': simulation,\
            'time': time,\
            'states': states
            })

    def simulate_ode_trace(self, time, stimulus):
        """Simulate deterministic ODE model under a given simulation"""
        return odeint(self._ode_system, self.initial_condition, time, args=(time, stimulus,))

    def simulate_ode_constant(self, t_end=1, stimulus=0, n_points=int(1e3)):
        """Simulate deterministic ODE model under a constant stimulus for a fixed time"""
        t = np.linspace(0, t_end, n_points).flatten()
        rates_q = self._update_q_matrix(stimulus)
        return odeint(lambda p, t, Q: np.dot(p,Q), self.initial_condition, t, args=(rates_q,))
        

### Model I/O ###

def export_model(data, filename):
    """Saves data dictionary in JSON format"""

    opts = jsbeautifier.default_options()
    opts.indent_size = 2
    data_beaut = jsbeautifier.beautify(json.dumps(data), opts)
    with open(filename, 'w') as file:
        file.write(data_beaut)

def open_model(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def validate_model(data):
    """Verify that all elements required for model specification are present"""

    required_keys = ['rate_constants','stim_dependence','initial_condition']
    
    for key in required_keys:
        assert key in data, f"KeyError: {key} must be provided with model specification."
    
    for key in required_keys:
        data[key] = np.asarray(data[key])
    
    for key in ['rate_constants','stim_dependence']:
        assert data[key].ndim == 2 and data[key].shape[0] == data[key].shape[1],\
            f"ValueError: {key} is not square. Shape is {data[key].shape}."
    
    while data['initial_condition'].ndim > 1:
        data['initial_condition'] = data['initial_condition'].flatten()
    
    for combo in combinations(required_keys, 2):
        assert data[combo[0]].shape[0] == data[combo[1]].shape[0],\
            f"ValueError: {combo[0]} with shape {data[combo[0]].shape[0]} does not match {combo[1]} with shape {data[combo[1]].shape[0]}."
    
    return data

def import_model(filename):
    """Imports data from JSON file into a MarkovModel object"""

    data = validate_model(open_model(filename))
    metadata = {key: value for key, value in data.items() if key not in ['rate_constants','stim_dependence','initial_condition']}
    model = MarkovModel(rate_constants=data['rate_constants'], stim_dependence=data['stim_dependence'], initial_condition=data['initial_condition'], metadata=metadata)
    
    return model

