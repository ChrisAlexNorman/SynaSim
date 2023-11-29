import numpy as np
import re


def extract_channel_opening(conductive_states, sample_path):

    channel_state = [(None, None)]
    for time, state in sample_path:

        if state[0] in conductive_states:
            toggle = 'Open'
        else:
            toggle = 'Close'
            
        if toggle != channel_state[-1][1]:
            channel_state.append((time, toggle))

    channel_state = channel_state[1:]
    toggle_times = [time for time, _ in channel_state]

    return channel_state, toggle_times


def get_output_values(toggle_times, output_times, initial_state):
    states = []
    current_state = initial_state
    toggle_index = 0
    toggle_len = len(toggle_times)
    
    for time in output_times:
        # Update toggle_index to the last toggle time that is less than or equal to the current time
        while toggle_index < toggle_len and toggle_times[toggle_index] <= time:
            current_state = 1 - current_state  # Flip the state
            toggle_index += 1
        
        states.append(current_state)
        
    return states


def extract_toggle_times(model, simulation):
    """Convert stochastic sample paths into lists of open / closed toggle times for channel conductivity (assuming closed initially)"""

    conductive_states = [state for state, conductive in zip(model.state_names, model.conductive) if conductive]
    toggle_times = []
    for sample_path in simulation['sample_paths']:
        _, times = extract_channel_opening(conductive_states, sample_path)
        toggle_times.append(times)
    
    return toggle_times


def piecewise_linear_segment(t_start, t_end, conductance):
    c1 =  np.interp(t_start, conductance['timestamp'], conductance['value'])
    c2 =  np.interp(t_end, conductance['timestamp'], conductance['value'])
    a = (c2 - c1) / (t_end - t_start)
    b = c1 - a * t_start
    return f"({a}*t+{b})"


def generate_piecewise_linear_string(toggle_times, conductance, initial_state):
    terms = []
    current_state = initial_state
    for i, t_start in enumerate(toggle_times):
        try:
            t_end = toggle_times[i + 1]
        except IndexError:
            t_end = conductance['timestamp'][-1]
        
        if current_state == 1:
            multiplier = piecewise_linear_segment(t_start, t_end, conductance)
            term = f"({multiplier}*(t>={t_start})*(t<{t_end}))"
            terms.append(term)
        
        # Flip the state for the next term
        current_state = 1 - current_state
    
    return " + ".join(terms)


def evaluate_piecewise_string(sample_times, function_str):
    pattern = r"\(([-\d\.]+)\*t\+([-\d\.]+)\)\*\(t>=([-\d\.]+)\)\*\(t<([-\d\.]+)\)"
    terms = re.findall(pattern, function_str)

    # Initialize an array to store the results
    results = np.zeros_like(sample_times)

    # Evaluate the piecewise function at each time point
    for term in terms:
        slope, intercept, lower_bound, upper_bound = map(float, term)
        mask = (sample_times >= lower_bound) & (sample_times < upper_bound)
        results[mask] = slope * sample_times[mask] + intercept

    return results