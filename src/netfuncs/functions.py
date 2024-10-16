def degree_distribution(G, number_of_bins=15, log_binning=True, density=True, directed=False):
    """
    Given a degree sequence, return the y values (probability) and the
    x values (support) of a degree distribution that you're going to plot.
    
    Parameters
    ----------
    G (nx.Graph):
        the network whose degree distribution to calculate

    number_of_bins (int):
        length of output vectors
    
    log_binning (bool):
        if you are plotting on a log-log axis, then this is useful
    
    density (bool):
        whether to return counts or probability density (default: True)
        Note: probability densities integrate to 1 but do not sum to 1. 

    directed (bool or str):
        if False, this assumes the network is undirected. Otherwise, the
        function requires an 'in' or 'out' as input, which will create the 
        in- or out-degree distributions, respectively.
        
    Returns
    -------
    bins_out, probs (np.ndarray):
        probability density if density=True node counts if density=False; binned edges
    
    """

    # Step 0: Do we want the directed or undirected degree distribution?
    if directed:
        if directed=='in':
            k = list(dict(G.in_degree()).values()) # get the in degree of each node
        elif directed=='out':
            k = list(dict(G.out_degree()).values()) # get the out degree of each node
        else:
            out_error = "Help! if directed!=False, the input needs to be either 'in' or 'out'"
            print(out_error)
            # Question: Is this the correct way to raise an error message in Python?
            #           See "raise" function...
            return out_error
    else:
        k = list(dict(G.degree()).values()) # get the degree of each node


    # Step 1: We will first need to define the support of our distribution
    kmax = np.max(k)    # get the maximum degree
    kmin = 0            # let's assume kmin must be 0


    # Step 2: Then we'll need to construct bins
    if log_binning:
        # array of bin edges including rightmost and leftmost
        bins = np.logspace(0, np.log10(kmax+1), number_of_bins+1)
    else:
        bins = np.linspace(0, kmax+1, num=number_of_bins+1)


    # Step 3: Then we can compute the histogram using numpy
    probs, _ = np.histogram(k, bins, density=density)


    # Step 4: Return not the "bins" but the midpoint between adjacent bin
    #         values. This is a better way to plot the distribution.
    bins_out = bins[1:] - np.diff(bins)/2.0
    
    return bins_out, probs

def all_shortest_from(G, node_i):
    """
    For a given node_i in the network, construct a dictionary containing
    the length of the shortest path between that node and all others in
    the network. Values of -1 correspond to nodes where no paths connect
    to node_i.
    
    Parameters
    ----------
    G (nx.Graph)
        the graph in question
    
    node_i (int or str)
        the label of the "source" node
    
    Returns
    -------
    distances (dict)
        dictionary where the key corresponds to other nodes in the network
        and the values indicate the shortest path length between that node
        and the original node_i source.
    
    """
    
    distances = {i: -1 for i in G.nodes()}

    # And distance of 0 from node_i to itself
    distances[node_i] = 0

    # Queue of nodes to visit next, starting with our source node_i
    queue = [node_i]

    # We want to:
    ### 1. Start with one element in our queue
    ### 2. Find its neighbors and add them to our queue
    ### 3. Select one of the neighbors from the queue
    ### 4. Add its neighbors to our queue
    ### 5. And so on, until the queue has been depleted

    # This is a great time for a `while` loop!

    # while there is still a queue of any positive length (instead of len(queue)>0)...
    while queue:
        # select oldest item in the list (as opposed to last item)
        current_node = queue.pop(0)

        # find all the 
        for next_node in G.neighbors(current_node):
            if distances[next_node] < 0:
                # set next_node's distance to be +1 whatever the distance to the current node is
                ### e.g. if current_node distance is 0, its neighbors will be 0+1 distance away
                ### ... and so on
                distances[next_node] = distances[current_node] + 1

                # and since we havent seen next_node before (distance still < 0)
                # we will need to append it to our queue
                queue.append(next_node)    
    
    return distances

def degree_preserving_randomization(G, n_iter=1000):
    """
    Perform degree-preserving randomization on a graph.

    Degree-preserving randomization, also known as edge swapping or rewiring, 
    is a method for creating randomized versions of a graph while preserving 
    the degree distribution of each node. This is achieved by repeatedly 
    swapping pairs of edges in the graph, ensuring that the degree (number of 
    edges connected) of each node remains unchanged. The result is a graph 
    with the same degree distribution but a randomized edge structure, which 
    can be used as a null model to compare with the original network.

    Parameters
    ----------
    G : networkx.Graph
        The input graph to be randomized. The graph can be directed or 
        undirected, but it must be simple (i.e., no self-loops or parallel edges).

    n_iter : int, optional (default=1000)
        The number of edge swap iterations to perform. A higher number of 
        iterations leads to more randomization, but the degree distribution 
        remains preserved. Typically, the number of iterations should be 
        proportional to the number of edges in the graph for sufficient 
        randomization.

    Returns
    -------
    G_random : networkx.Graph
        A randomized graph with the same degree distribution as the original 
        graph `G`, but with a shuffled edge structure.

    Notes
    -----
    - This method works by selecting two edges at random, say (u, v) and (x, y), 
      and attempting to swap them to (u, y) and (x, v) (or (u, x) and (v, y)), 
      ensuring that no self-loops or parallel edges are created in the process.
    - Degree-preserving randomization is particularly useful for creating null 
      models in network analysis, as it allows for the investigation of whether 
      specific network properties (e.g., clustering, path lengths) are a result 
      of the network's structure or just its degree distribution.
    - The effectiveness of randomization depends on the number of iterations 
      (`n_iter`). As a rule of thumb, using about 10 times the number of edges 
      in the graph for `n_iter` often provides sufficient randomization.
    
    Example
    -------
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(10, 0.5)
    >>> G_random = degree_preserving_randomization(G, n_iter=100)
    
    Citations
    ---------
    Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D., & Alon, U. (2002). 
    Network motifs: simple building blocks of complex networks. *Science*, 298(5594), 824-827.
    
    Maslov, S., & Sneppen, K. (2002). Specificity and stability in topology of protein networks. 
    *Science*, 296(5569), 910-913.
    """

    G_random = G.copy()
    edges = list(G_random.edges())
    num_edges = len(edges)

    for _ in range(n_iter):
        # Select two random edges (u, v) and (x, y)
        edge1_id = np.random.choice(list(range(len(edges))))
        u, v = edges[edge1_id]
        edge2_id = np.random.choice(list(range(len(edges))))
        x, y = edges[edge2_id]

        # Avoid selecting the same edge pair or creating self-loops
        if len({u, v, x, y}) == 4:
            # Swap the edges with some probability
            if np.random.rand() > 0.5:
                # Swap (u, v) with (u, y) and (x, v)
                if not (G_random.has_edge(u, y) or G_random.has_edge(x, v)):
                    G_random.remove_edge(u, v)
                    G_random.remove_edge(x, y)
                    G_random.add_edge(u, y)
                    G_random.add_edge(x, v)
            else:
                # Swap (u, v) with (u, x) and (v, y)
                if not (G_random.has_edge(u, x) or G_random.has_edge(v, y)):
                    G_random.remove_edge(u, v)
                    G_random.remove_edge(x, y)
                    G_random.add_edge(u, x)
                    G_random.add_edge(v, y)

        # Update edge list after changes
        edges = list(G_random.edges())


    return G_random

def bfs(explore_queue, nodes_visited, graph):
    if len(explore_queue) == 0:
        return nodes_visited
    else:
        current_node = explore_queue.pop(0)
        print('visiting node ' + str(current_node))
        for neighbor in G.neighbors(current_node):
            if neighbor in nodes_visited:
                continue
            else:
                nodes_visited[neighbor] = nodes_visited[current_node] + 1
                explore_queue.append(neighbor)
        return bfs(explore_queue, nodes_visited, graph)

def dfs(explore_stack, nodes_visited, graph):
    if len(explore_stack) == 0:
        return nodes_visited
    else:
        current_node = explore_stack.pop(-1)
        print('visiting node {}'.format(str(current_node)))
        for neighbor in G.neighbors(current_node):
            if neighbor in nodes_visited:
                continue
            else:
                nodes_visited[neighbor] = nodes_visited[current_node] + 1
                explore_stack.append(neighbor)
        return dfs(explore_stack, nodes_visited, graph)

def calculate_modularity(G, partition):
    """
    Calculates the modularity score for a given partition of the graph, whether the graph is weighted or unweighted.
    
    Modularity is a measure of the strength of division of a network into communities. It compares the actual 
    density of edges within communities to the expected density if edges were distributed randomly. For weighted 
    graphs, the weight of the edges is taken into account.

    The modularity Q is calculated as:
    
    Q = (1 / 2m) * sum((A_ij - (k_i * k_j) / (2m)) * delta(c_i, c_j))

    where:
    - A_ij is the weight of the edge between nodes i and j (1 if unweighted).
    - k_i is the degree of node i (or the weighted degree for weighted graphs).
    - m is the total number of edges in the graph, or the total weight of the edges if the graph is weighted.
    - delta(c_i, c_j) is 1 if nodes i and j belong to the same community, and 0 otherwise.

    Parameters:
    -----------
    G : networkx.Graph
        The input graph, which can be undirected and either weighted or unweighted. The graph's nodes represent the 
        entities, and its edges represent connections between them.
    
    partition : list of sets
        A list of sets where each set represents a community. Each set contains the nodes belonging to that community. 
        For example, [{0, 1, 2}, {3, 4}] represents two communities, one with nodes 0, 1, and 2, and another with nodes 
        3 and 4.
    
    Returns:
    --------
    float
        The modularity score for the given partition of the graph. A higher score indicates stronger community structure, 
        and a lower (or negative) score suggests weak or no community structure.

    Notes:
    ------
    - If the graph has weights, they will be used in the modularity calculation. If no weights are present, the function 
      assumes each edge has a weight of 1 (i.e., unweighted).
    
    - The function assumes that all nodes in the graph are assigned to exactly one community. If any node is missing 
      from the community list, it is treated as not belonging to any community, and the results may not be accurate.
    
    - If the graph has no edges, the modularity is undefined, and this function will return 0 because the total number 
      of edges (2m) would be zero.
    
    Example:
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> communities = [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}]
    >>> modularity_score = calculate_modularity(G, communities)
    >>> print("Modularity:", modularity_score)
    
    References:
    -----------
    Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community structure 
    in networks. Physical Review E, 69(2), 026113.
    """
  
    def remap_partition(partition):
        """
        Converts and remaps a partition to a list-of-lists structure suitable for modularity calculations.

        This function remaps the input partition (whether it's in dictionary form or a flat list of community labels) 
        to a list-of-lists format, where each list represents a community and contains the nodes in that community. 
        The function also ensures that community labels are contiguous integers starting from 0, which is typically 
        required for modularity-based algorithms.
        """

        # if partition is a dictionary where the keys are nodes and values communities
        if type(partition)==dict:
            unique_comms = np.unique(list(partition.values()))
            comm_mapping = {i:ix for ix,i in enumerate(unique_comms)}
            for i,j in partition.items():
                partition[i] = comm_mapping[j]

            unique_comms = np.unique(list(partition.values()))
            communities = [[] for i in unique_comms]
            for i,j in partition.items():
                communities[j].append(i)
                
            return communities

        # if partition is a list of community assignments
        elif type(partition)==list and\
                not any(isinstance(el, list) for el in partition):
            unique_comms = np.unique(partition)
            comm_mapping = {i:ix for ix,i in enumerate(unique_comms)}
            for i,j in enumerate(partition):
                partition[i] = comm_mapping[j]

            unique_comms = np.unique(partition)
            communities = [[] for i in np.unique(partition)]
            for i,j in enumerate(partition):
                communities[j].append(i)

            return communities

        # otherwise assume input is a properly-formatted list of lists
        else:
            communities = partition.copy()
            return communities


    # We now should have a list-of-lists structure for communities
    communities = remap_partition(partition)
    
    # Total weight of edges in the graph (or number of edges if unweighted)
    if nx.is_weighted(G):
        m = G.size(weight='weight')
        degree = dict(G.degree(weight='weight'))  # Weighted degree for each node
    else:
        m = G.number_of_edges()  # Number of edges in the graph
        degree = dict(G.degree())  # Degree for each node (unweighted)

    # Modularity score
    modularity_score = 0.0
    
    # Loop over all pairs of nodes i, j within the same community
    for community in communities:
        for i in community:
            for j in community:
                # Get the weight of the edge between i and j, or assume weight 1 if unweighted
                if G.has_edge(i, j):
                    A_ij = G[i][j].get('weight', 1)  # Use weight if available, otherwise assume 1
                else:
                    A_ij = 0  # No edge between i and j

                # Expected number of edges (or weighted edges) between i and j in a random graph
                expected_edges = (degree[i] * degree[j]) / (2 * m)

                # Contribution to modularity
                modularity_score += (A_ij - expected_edges)

    # Normalize by the total number of edges (or total edge weight) 2m
    modularity_score /= (2 * m)


    return modularity_score

def girvan_newman(H, output='dict'):
    """
    Implements the Girvan-Newman algorithm for detecting communities in a graph.
    
    The Girvan-Newman method works by iteratively removing edges from the graph 
    based on their edge betweenness centrality, which is a measure of the frequency 
    with which edges appear on the shortest paths between pairs of nodes. As edges 
    with high centrality values are removed, the graph breaks down into smaller 
    connected components, each representing a community.
    
    Parameters:
    -----------
    H : networkx.Graph
        The input graph on which to perform community detection.
    
    output : strong, optional (default='dict')
        'dict'  - the algorithm will return a dictionary where the keys correspond to
                  nodes and values are the community label
        'lists' - a list of lists of length n_comms will be returned, such that the ith
                  element of the list contains a nodelist of nodes that are assigned to
                  community i
        'list'  - a list of community assignments, in the same order as G.nodes()
    
    Returns:
    --------
    dict:
        A dictionary or similar object where the keys are node identifiers and
        the values are  community labels (integers), indicating which
        community each node belongs to.
    
    
    Notes:
    ------
    - Modularity is a measure of the strength of the division of a network into 
      communities. Higher values indicate stronger community structure.
    
    - The Girvan-Newman algorithm is computationally expensive (O(n*m^2), where n 
      is the number of nodes and m is the number of edges), making it impractical 
      for very large graphs.
    
    Example:
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> partition = girvan_newman(G)
    >>> for node, community in partition.items():
    >>>     print(f"Node {node} belongs to community {community}")
    """

    # Make a copy of the graph so we don't modify the original graph
    G = H.copy()

    # Initialize variables to track the best partition and modularity
    best_partition = None
    best_modularity = -1

    # Keep removing edges until all components are disconnected
    while G.number_of_edges() > 0:
        # Compute edge betweenness centrality for all edges
        edge_betweenness = nx.edge_betweenness_centrality(G)

        # Find the edge with the highest betweenness
        max_edge = max(edge_betweenness, key=edge_betweenness.get)

        # Remove the edge with the highest betweenness
        G.remove_edge(*max_edge)

        # Find connected components (each component is a community)
        components = list(nx.connected_components(G))

        # Assign community labels to nodes
        community_dict = {}
        for i, component in enumerate(components):
            for node in component:
                community_dict[node] = i

        # Compute modularity for the current partition on the original graph
        try:
            current_modularity = modularity(H, components)
        except ZeroDivisionError:
            # If division by zero occurs, set modularity to 0
            current_modularity = 0

        # If this modularity is better than the best so far, store it
        if current_modularity > best_modularity:
            best_modularity = current_modularity
            best_partition = community_dict.copy()

    best_partition = {i:best_partition[i] for i in G.nodes()}
    # Return the best partition
    if output == 'dict':
        return best_partition

    if output == 'list':
        return list(best_partition.values())
        
    if output == 'lists':
        comm_out = [[] for i in np.unique(list(best_partition.values()))]
        for i,j in best_partition.items():
            comm_out[j].append(i)
        return comm_out

    import graph_tool.all as gt

def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key).

    This function is adapted from Benjamin Bengfort's blog post!
    https://bbengfort.github.io/2016/06/graph-tool-from-networkx/
    """
    if isinstance(key, str):  # Keep the key as a string
        key = key  # No encoding necessary in Python 3

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, str):
        tname = 'string'
        
    elif isinstance(value, bytes):
        tname = 'bytes'

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


# def nx2gt(nxG):
#     """
#     Converts a networkx graph to a graph-tool graph.
#     """
#     # Phase 0: Create a directed or undirected graph-tool Graph
#     gtG = gt.Graph(directed=nxG.is_directed())

#     # Add the Graph properties as "internal properties"
#     for key, value in nxG.graph.items():
#         # Convert the value and key into a type for graph-tool
#         tname, value, key = get_prop_type(value, key)

#         prop = gtG.new_graph_property(tname) # Create the PropertyMap
#         gtG.graph_properties[key] = prop     # Set the PropertyMap
#         gtG.graph_properties[key] = value    # Set the actual value

#     # Phase 1: Add the vertex and edge property maps
#     # Go through all nodes and edges and add seen properties
#     # Add the node properties first
#     nprops = set() # cache keys to only add properties once
#     for node, data in nxG.nodes(data=True):

#         # Go through all the properties if not seen and add them.
#         for key, val in data.items():
#             if key in nprops: continue # Skip properties already added

#             # Convert the value and key into a type for graph-tool
#             tname, _, key  = get_prop_type(val, key)

#             prop = gtG.new_vertex_property(tname) # Create the PropertyMap
#             gtG.vertex_properties[key] = prop     # Set the PropertyMap

#             # Add the key to the already seen properties
#             nprops.add(key)

#     # Also add the node id: in NetworkX a node can be any hashable type, but
#     # in graph-tool node are defined as indices. So we capture any strings
#     # in a special PropertyMap called 'id' -- modify as needed!
#     gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

#     # Add the edge properties second
#     eprops = set() # cache keys to only add properties once
#     for src, dst, data in nxG.edges(data=True):

#         # Go through all the edge properties if not seen and add them.
#         for key, val in data.items():
#             if key in eprops: continue # Skip properties already added

#             # Convert the value and key into a type for graph-tool
#             tname, _, key = get_prop_type(val, key)

#             prop = gtG.new_edge_property(tname) # Create the PropertyMap
#             gtG.edge_properties[key] = prop     # Set the PropertyMap

#             # Add the key to the already seen properties
#             eprops.add(key)

#     # Phase 2: Actually add all the nodes and vertices with their properties
#     # Add the nodes
#     vertices = {} # vertex mapping for tracking edges later
#     for node, data in nxG.nodes(data=True):

#         # Create the vertex and annotate for our edges later
#         v = gtG.add_vertex()
#         vertices[node] = v

#         # Set the vertex properties, not forgetting the id property
#         data['id'] = str(node)
#         for key, value in data.items():
#             gtG.vp[key][v] = value # vp is short for vertex_properties

#     # Add the edges
#     for src, dst, data in nxG.edges(data=True):

#         # Look up the vertex structs from our vertices mapping and add edge.
#         e = gtG.add_edge(vertices[src], vertices[dst])

#         # Add the edge properties
#         for key, value in data.items():
#             gtG.ep[key][e] = value # ep is short for edge_properties

#     # Done, finally!
#     return gtG