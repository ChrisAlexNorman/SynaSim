import numpy as np
import numexpr as ne
import json
import jsbeautifier
from scipy.integrate import odeint

class MarkovModel:

    def __init__(self, rates=np.array([0]), parameters={}, initial_condition=np.array([1]), metadata={}, multiprocessing=True):
        self.rates = rates
        self.parameters = parameters
        self.parameter_values = {name: info['value'] for name, info in parameters.items()}
        self.n_states = self.rates.shape[0]
        self.initial_condition = initial_condition
        self.metadata = self._validate_metadata(metadata)
        self.simulations = []
        self.multiprocessing = multiprocessing

    def __str__(self):
        return self.metadata['name'] + f": {self.n_states}-state Markov chain"

    def __len__(self):
        return self.n_states

    def _validate_metadata(self, metadata):
        if 'name' not in metadata:
            metadata['name'] = 'Unnamed'
        if 'state_names' not in metadata:
            metadata['state_names'] = ['S'+str(n) for n in range(0,self.n_states)]
        return metadata

    def _get_q_matrix(self, rates, params):
        q_matrix = np.asarray([[ne.evaluate(str(element), params).item() for element in row] for row in rates])
        q_matrix[np.diag_indices(len(rates))] = -np.sum(q_matrix, axis=1)
        return q_matrix

    def _ode_system(self, states, t, stimuli):
        """Return dp/dt at time t for time-dependent transition rates"""
        stimuli_at_t = {name: np.interp(t, stimulus['time'], stimulus['stim']) for  name, stimulus in stimuli.items()}
        params_at_t = {**self.parameter_values, **stimuli_at_t}
        q_matrix_at_t = self._get_q_matrix(self.rates, params_at_t)
        return np.dot(states, q_matrix_at_t)

    def simulate_ode_trace(self, stimuli):
        """Simulate deterministic ODE model under a given simulation"""
        time = next(iter(stimuli.items()))[1]['time']
        return odeint(self._ode_system, self.initial_condition, time, args=(stimuli,))

    def run_simulations(self, simulations=[]):
        """Run each simulation in simulations."""
        self.simulations = [{'stimuli': stimuli, 'states': self.simulate_ode_trace(stimuli)} for stimuli in simulations]
        return self.simulations

    def clear_simulations(self):
        simulations = self.simulations
        self.simulations = []
        return simulations

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
    for key in ['rates', 'parameters', 'initial_condition']:
        assert key in data, f"KeyError: {key} must be provided with model specification."
    assert data['rates'].ndim == 2 and data['rates'].shape[0] == data['rates'].shape[1], f"ValueError: {key} is not square. Shape is {data[key].shape}."
    assert len(data['rates']) == len(data['initial_condition']), f"Rate matrix describes {len(data['rates'])} states but initial condition specifies {len(data['initial_condition'])}"
    return data

def import_model(filename):
    """Imports data from JSON file into a MarkovModel object"""
    data = validate_model(open_model(filename))
    metadata = {key: value for key, value in data.items() if key not in ['rates','parameters','initial_condition']}
    return MarkovModel(rates=data['rates'], parameters=data['parameters'], initial_condition=data['initial_condition'], metadata=metadata)
