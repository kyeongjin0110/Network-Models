In this homework, you are asked to generate networks based on different models. By default, set the
number of nodes as n=1000. But if that is too much on your machine, you can change it to a smaller
(but still sufficiently large) value.

1. Erdos-Renyi random graph model
- Generate a random network with n nodes using the Erdos-Renyi model with an edge probability p.
- Check the number of connected components with respect to the value of p.
- For two different values of p, obtain the degree distributions, clustering coefficients, and diameters.
Compare them with the theoretical values. If you observe any discrepancy between the theoretical
and obtained values, analyze possible reasons.

2. Barabasi-Albert model
- Generate a network with n nodes using the Barbasi-Albert model for c=4.Use a complete graph with
4 nodes as a starting network, and add a node at a time based on the preferential attachment principle.
- Measure the degree distributions at intermediate steps (e.g., when the network has 100, 1000, and
10000 nodes). Fit each of them to a power-law distribution to obtain the value of the degree exponent.
Does it converge to the theoretical value?
- Examine the clustering coefficient and diameter at the end. Compare them with the theoretical values,
and discuss any discrepancy.

3. Watts-Strogatz model
- Generate a Watts-Strogatz network with a shortcut probability of p, starting from a circle model with
n nodes and a degree of c=4 for each node. Use the modified version where edges in the circle model
are not removed when shortcuts are added. Try different values of p.
- At intermediate steps, obtain the degree distributions, clustering coefficients, and diameters. Discuss
the trends. Do they follow the expected theoretical trends?