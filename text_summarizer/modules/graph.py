# ============================================================
# GRAPH GENERATION MODULE
# ============================================================

import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict


def build_entity_graph(entities: List[Dict]) -> nx.Graph:
    G = nx.Graph()
    
    for i in range(len(entities)-1):
        e1 = entities[i]["text"]
        e2 = entities[i+1]["text"]
        G.add_edge(e1, e2)
    
    return G


def visualize_graph(G: nx.Graph):
    pos = nx.spring_layout(G)
    
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=1),
                           hoverinfo='none',
                           mode='lines')
    
    node_x = []
    node_y = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center"
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    return fig