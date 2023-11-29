import os
import random
import numpy as np
import pandas as pd
import numexpr as ne
import xml.etree.ElementTree as ET
from scipy import integrate
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChannelRandomiser:

    def __init__(self, az_z, clusters=None, n_channels_by_type=None, channel_types=None, libraries=None):
        self.az_z = az_z
        self.clusters = clusters or []
        self.n_channels_by_type = n_channels_by_type or []
        self.channel_types = channel_types or []
        self.libraries = libraries or []

    def rand_channel_location(self, cluster_dims):
        angle = random.uniform(0, 2*np.pi)
        radius = random.uniform(0, cluster_dims[2])
        return [cluster_dims[0] + radius * np.cos(angle), cluster_dims[1] + radius * np.sin(angle)]

    def rand_curr_from_lib(self, library):
        currents = pd.read_csv(library, header=None, names=['current'])
        currents = currents.fillna('0.0')
        rand_index = random.randint(0, len(currents)-1)
        return rand_index, currents.iloc[rand_index]['current']

    def randomise_channels(self):
        df_list = []
        for idx_cluster, cluster_dims in enumerate(self.clusters):
            for idx_type, n_channels in enumerate(self.n_channels_by_type):
                channel_type = self.channel_types[idx_type]
                for idx_channel in range(n_channels):
                    x, y = self.rand_channel_location(cluster_dims)
                    curr_idx, curr_str = self.rand_curr_from_lib(self.libraries[idx_type])

                    df_list.append({
                        'Cluster Index' : idx_cluster+1,
                        'Channel Name' : f"CaChC{idx_cluster+1}{channel_type}{idx_channel+1}_Membrane",
                        'Channel Type' : channel_type,
                        'x (um)' : x,
                        'y (um)' : y,
                        'z (um)' : self.az_z,
                        'Current Index' : curr_idx,
                        'Current String' : curr_str
                    })
        
        self.data = pd.DataFrame.from_records(df_list)
        self.data[['Cluster Index', 'Current Index']] = self.data[['Cluster Index', 'Current Index']].astype(int)
        return self.data
    
class ClusterCurrent:

    def __init__(self, channel_data, t, func_str):
        self.channel_data = channel_data
        self.t = t
        self.func_str = func_str
        self.aggregate_current()
        self.fit_cluster_current()

    def aggregate_current(self):
        t = self.t
        N = F = d = 1
        scaling_factor = N / (2 * F * d ** 2)
        self.total_current = [0] * len(t)
        for index, row in self.channel_data.iterrows():
            channel_curr = ne.evaluate(row['Current String'])
            if isinstance(channel_curr, float):
                channel_curr = np.asarray([channel_curr] * len(t))
            self.total_current += channel_curr
        self.total_current /= scaling_factor
        return self.total_current

    def fit_cluster_current(self):
        total_current = self.total_current
        t = self.t
        func_str = self.func_str

        def objective(params):
            a, b, x0 = params
            diff = integrate.cumtrapz(ne.evaluate(func_str), t) - integrate.cumtrapz(total_current, t)
            return np.sum(diff ** 2)

        initial_params = [1e-3, 0.1, 1e-3]
        bounds = [(1e-4, 1e-2), (1e-4, 1), (1e-4, 1e-2)]
        self.func_fit = minimize(objective, initial_params, bounds=bounds, tol=1e-15)
        a, b, x0 = self.func_fit.x
        self.best_fit = ne.evaluate(func_str)
        return self.func_fit

class XMLEditor:

    def __init__(self, master_xml, master_tree, namespace, master_name, project_name, project_data):
        self.master_xml = master_xml
        self.master_tree = master_tree
        self.master_name = master_name
        self.namespace = namespace
        self.project_name = project_name
        self.project_data = project_data
        self.project_xml = self.master_xml.replace(self.master_name, self.project_name)
    
    def update_channels(self):
        root = self.master_tree.getroot()
        for index, row in self.project_data.iterrows():
            channel_name = row['Channel Name']

            # Update x position
            x_name = channel_name.replace('CaCh','x').replace('_Membrane','')
            old_x_position = root.find(f".//vcml:Parameter[@Name='{x_name}']", self.namespace).text
            old_x_string = f'<Parameter Name="{x_name}" Role="user defined" Unit="um">{old_x_position}</Parameter>'
            new_x_position = row['x (um)']
            new_x_string = f'<Parameter Name="{x_name}" Role="user defined" Unit="um">{new_x_position}</Parameter>'
            self.project_xml = self.project_xml.replace(old_x_string, new_x_string)

            # Update y position
            y_name = x_name.replace('x','y')
            old_y_position = root.find(f".//vcml:Parameter[@Name='{y_name}']", self.namespace).text
            old_y_string = f'<Parameter Name="{y_name}" Role="user defined" Unit="um">{old_y_position}</Parameter>'
            new_y_position = row['y (um)']
            new_y_string = f'<Parameter Name="{y_name}" Role="user defined" Unit="um">{new_y_position}</Parameter>'
            self.project_xml = self.project_xml.replace(old_y_string, new_y_string)
            
            # Update current
            reaction_name = f"Cluster{row['Cluster Index']}_{row['Channel Type']}{x_name[-1]}"
            old_current_string = root.find(f".//vcml:FluxStep[@Name='{reaction_name}']", self.namespace).find(".//vcml:Parameter[@Name='J']", self.namespace).text.replace('>','&gt;').replace('<','&lt;')
            new_current_string = f"{channel_name} * {row['Current String']}".replace('>','&gt;').replace('<','&lt;')
            self.project_xml = self.project_xml.replace(old_current_string, new_current_string)

        return self.project_xml
    
    def update_clusters(self):
        root = self.master_tree.getroot()
        # NOTE: clusters appear in REVERSE ORDER in XML file. i.e. CaCh02 ... CaCh01
        reversed_data = self.project_data[::-1]
        for index, row in enumerate(reversed_data):
            # Update current parameters
            for param, old_value in [('a', '9.2246E-4'), ('b', '0.178'), ('x0', '8.036E-4')]:
                old_string = f'<Parameter Name="{param}" Role="user defined" Unit="tbd">{old_value}</Parameter>'
                new_string = f'<Parameter Name="{param}" Role="user defined" Unit="tbd">{row[param]}</Parameter>'
                self.project_xml = self.project_xml.replace(old_string, new_string, 1)

        return self.project_xml
        
def generate(master_file_channels, master_file_clusters, az_z, sim_time, project_base_name='', export_path='', n_sims=1, clusters=None, channel_types=None, n_channels_by_type=None, libraries=None):
    all_currs = pd.DataFrame()
    stoch_currs = []
    fit_currs = []
    t = np.linspace(0, sim_time, 1000)
    log_normal_str = 'a*exp(-0.5*((log((1e-10+t)/x0))/b)**2)/(1e-10+t)'

    with open(master_file_channels, 'r') as file:
        master_xml_indiv = file.read()
    master_tree_indiv = ET.parse(master_file_channels)
    master_name_indiv = master_tree_indiv.getroot()[0].attrib['Name']
    with open(master_file_clusters, 'r') as file:
        master_xml_clust = file.read()
    master_tree_clust = ET.parse(master_file_clusters)
    master_name_clust = master_tree_clust.getroot()[0].attrib['Name']
    namespace = {'vcml': 'http://sourceforge.net/projects/vcell/vcml'}

    df_list = []
    for sim in range(1, n_sims+1):
        
        # Randomise channel locations and currents
        channels_df = ChannelRandomiser(az_z=az_z,
                                        clusters=clusters,
                                        n_channels_by_type=n_channels_by_type,
                                        channel_types=channel_types,
                                        libraries=libraries).randomise_channels()
        all_currs = pd.concat([all_currs, channels_df], ignore_index=True)
        channels_df.to_csv(os.path.join(export_path, f"{project_base_name}_channels_sim-{sim}.csv"), index=False)
                                        
        # Plot channel locations
        fig = plot_channel_locations(channels_df, az_rad=np.sqrt(0.15/np.pi), clusters=clusters)
        fig.update_layout(title=f'Channel Locations - Sim {sim}')
        fig.show()

        # Export to new XML
        project_name = f"{project_base_name}_channels_sim-{sim}"
        editor = XMLEditor(master_xml_indiv, master_tree_indiv, namespace, master_name_indiv, project_name, project_data=channels_df)
        editor.update_channels()
        project_file = os.path.join(export_path, f"{project_base_name}_channels_sim-{sim}.vcml")
        with open(project_file, 'wt', encoding='utf-8') as file:
            file.write(editor.project_xml)
        
        # Fit log-normal to cluster current
        clust_list = []
        for idx_cluster in channels_df['Cluster Index'].unique():
            cluster = ClusterCurrent(channel_data=channels_df[channels_df['Cluster Index'] == idx_cluster], t=t, func_str=log_normal_str)
            a, b, x0 = cluster.func_fit.x
            sse = cluster.func_fit.fun
            clust_list.append({
                'Sim Index' : sim,
                'Cluster Index' : idx_cluster,
                'a' : a,
                'b' : b,
                'x0' : x0,
                'SSE (Integral)' : sse
            })
            stoch_currs.append(cluster.total_current)
            fit_currs.append(cluster.best_fit)
        df_list.extend(clust_list)

        # Export to new XML
        project_name = f"{project_base_name}_clusters_sim-{sim}"
        editor = XMLEditor(master_xml_clust, master_tree_clust, namespace, master_name_clust, project_name, project_data=clust_list)
        editor.update_clusters()
        project_file = os.path.join(export_path, f"{project_base_name}_clusters_sim-{sim}.vcml")
        with open(project_file,'wt', encoding='utf-8') as file:
            file.write(editor.project_xml)
    
    # Show all channel locations across simulations
    fig = plot_channel_locations(all_currs, az_rad=np.sqrt(0.15/np.pi), clusters=clusters)
    fig.update_layout(title=f'Channel Locations - All {n_sims} Sims')
    fig.show()
    fig = compare_current_fits(t, stoch_currs, fit_currs)
    fig.show()

    cluster_df = pd.DataFrame.from_records(df_list)
    cluster_df[['Sim Index', 'Cluster Index']] = cluster_df[['Sim Index', 'Cluster Index']].astype(int)
    cluster_df.to_csv(os.path.join(export_path, f"{project_base_name}_clusters_all_sims.csv"), index=False)


def analyse(project_path, clusters=None):
    '''Summarise the results of previously generated project files'''
    clusters_file = None
    channels_files = []

    # List all files in the directory
    for file in os.listdir(project_path):
        if 'channels' in file:
            channels_files.append(file)
        elif 'clusters' in file:
            clusters_file = file

    n_sims = len(channels_files)
    all_currs = pd.DataFrame()
    stoch_currs = []
    fit_currs = []
    t = np.linspace(0, 2e-3, 1000)
    log_normal_str = 'a*exp(-0.5*((log((1e-10+t)/x0))/b)**2)/(1e-10+t)'

    cluster_df = pd.read_csv(clusters_file)

    for idx, channels_file in enumerate(channels_files):
        sim = idx + 1
        channels_df = pd.read_csv(channels_file)
        all_currs = pd.concat([all_currs, channels_df], ignore_index=True)
        fig = plot_channel_locations(channels_df, az_rad=np.sqrt(0.15/np.pi), clusters=clusters)
        fig.update_layout(title=f'Channel Locations - Sim {sim}')
        fig.show()

        for idx_cluster in channels_df['Cluster Index'].unique():
            cluster = ClusterCurrent(channel_data=channels_df[channels_df['Cluster Index'] == idx_cluster], t=t, func_str=log_normal_str)
            stoch_currs.append(cluster.total_current)
            a, b, x0 = cluster_df[['a', 'b', 'x0']][(cluster_df['Sim Index'] == sim) & (cluster_df['Cluster Index'] == idx_cluster)].values[0]
            fit_currs.append(ne.evaluate(log_normal_str))

    fig = plot_channel_locations(all_currs, az_rad=np.sqrt(0.15/np.pi), clusters=clusters)
    fig.update_layout(title=f'Channel Locations - All {n_sims} Sims')
    fig.show()
    fig = compare_current_fits(t, stoch_currs, fit_currs)
    fig.show()

    return all_currs, stoch_currs, fit_currs


### VISUALISATION ###


def compare_current_fits(time, stoch_currs, fit_currs):
    # Assume entries are valid

    if not isinstance(stoch_currs, list):
        return compare_current_fit_single(time, stoch_currs, fit_currs)

    n_cols = 3
    n_rows = int(np.ceil(len(stoch_currs)/n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols)

    row = col = 1
    for idx_curr in range(len(stoch_currs)):
        for trace in compare_current_fit_single(time, stoch_currs[idx_curr], fit_currs[idx_curr]).data:
            if idx_curr > 0:
                trace.showlegend=False
            fig.add_trace(trace, row=row, col=col)
            fig.update_xaxes(ticks='inside', linecolor='black', row=row, col=col)
            fig.update_yaxes(ticks='inside', linecolor='black', row=row, col=col)

        if col == n_cols:
            col = 1
            row += 1
        else:
            col += 1
        
    for col in range(1, n_cols+1):
        fig.update_xaxes(title_text = 'Time (s)', row=n_rows, col=col)
    for row in range(1, n_rows+1):
        fig.update_yaxes(title_text = 'Current (pA)', row=row, col=1)
    fig.update_layout(hovermode='x', plot_bgcolor='white', width=1200, height=1200)
    
    return fig


def compare_current_fit_single(time, stoch_curr, fit_curr):

    layout = go.Layout(title='',
                       showlegend=True,
                       width=800, height=1600,
                       hovermode='x',
                       xaxis={'title' : 'Time (s)','showgrid' : True, 'zeroline' : True},
                       yaxis={'title' : 'Current (pA)', 'showgrid' : True, 'zeroline' : True})
    
    traces = [
        go.Scatter(x=time, y=stoch_curr, mode='markers', marker={'color':'black'}, name='Stochastic'),
        go.Scatter(x=time, y=fit_curr, mode='lines', line={'color':'red'}, name='Analytic')
        ]
    
    return go.Figure(data=traces, layout=layout)


def plot_channel_locations(df, channel_rad=5, az_rad=0, clusters=[]):
    colors_on = ['rgba(0, 255, 0, 0.9)', 'rgba(0, 0, 255, 0.9)', 'rgba(255, 0, 0, 0.9)']
    colors_off = ['rgba(0, 85, 0, 0.9)', 'rgba(0, 0, 85, 0.9)', 'rgba(85, 0, 0, 0.9)']

    traces = []
    for idx_type, type in enumerate(df['Channel Type'].unique()):

        channels_off = df[['x (um)', 'y (um)']][(df['Channel Type'] == type) & (df['Current String'] == '0.0')]
        scatter_off = go.Scatter(x=channels_off['x (um)'], y=channels_off['y (um)'], mode='markers',
                             marker={'symbol' : 'square',
                                     'color' : colors_off[idx_type],
                                     'size' : channel_rad,
                                     'sizemode' : 'diameter'
                                     },
                             name=f"{type} Inactive")
        traces.append(scatter_off)

        channels_on = df[['x (um)', 'y (um)']][(df['Channel Type'] == type) & (df['Current String'] != '0.0')]
        scatter_on = go.Scatter(x=channels_on['x (um)'], y=channels_on['y (um)'], mode='markers',
                             marker={'symbol' : 'square',
                                     'color' : colors_on[idx_type],
                                     'size' : channel_rad,
                                     'sizemode' : 'diameter',
                                     'line' : {
                                         'color' : 'yellow',
                                         'width' : 0
                                     }
                                     },
                             name=f"{type} Active")
        traces.append(scatter_on)
    
    theta = np.linspace(0, 2*np.pi, 100)
    az_x = az_rad * np.cos(theta)
    az_y = az_rad * np.sin(theta)
    traces.append(go.Scatter(x=az_x, y=az_y, mode='lines', line={'color' : 'black'}, name='AZ', showlegend=False))

    for cluster in clusters:
        x = cluster[0] + cluster[2] * np.cos(theta)
        y = cluster[1] + cluster[2] * np.sin(theta)
        traces.append(go.Scatter(x=x, y=y, mode='lines', line={'color' : 'black'}, name='Cluster', showlegend=False))

    layout = go.Layout(title='Channel Locations', showlegend=True,
                       width=600, height=600,
                       xaxis={'showgrid' : False, 'zeroline' : False, 'title' : 'x (um)'},
                       yaxis={'showgrid' : False, 'zeroline' : False, 'title' : 'y (um)'})
    return go.Figure(data=traces, layout=layout)
    