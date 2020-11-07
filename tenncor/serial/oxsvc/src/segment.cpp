#include "tenncor/serial/oxsvc/segment.hpp"

#ifdef DISTR_OX_SEGMENT_HPP

namespace segment
{

// Floyd-Warshall to model and detect minimum path/distance between nodes
void floyd_warshall (std::vector<size_t>& distance,
	types::StrUMapT<size_t>& vertices, const distr::ox::GraphT& nodes)
{
	size_t n = nodes.size();
	// min distance between 2 nodes can be no greater than n - 1
	distance = std::vector<size_t>(n * n, n);
	vertices.reserve(n);
	for (auto& node : nodes)
	{
		vertices.emplace(node.first, vertices.size());
	}

	auto dist = sqr_curry(distance, n);
	for(auto& vert : vertices)
	{
		size_t u = vert.second;
		auto& edges = nodes.at(vert.first)->edges_;
		for (auto& edge : edges)
		{
			size_t v = vertices.at(edge->get_name());
			dist(u, v) = dist(v, u) = 1;
		}
		dist(u, u) = 0;
	}
	for (size_t k = 0; k < n; ++k)
	{
		for (size_t i = 0; i < n; ++i)
		{
			for (size_t j = 0; j < n; ++j)
			{
				if (dist(i, j) > dist(i, k) + dist(k, j))
				{
					dist(i, j) = dist(i, k) + dist(k, j);
				}
			}
		}
	}
}

std::vector<distr::ox::GraphT> disjoint_graphs (const distr::ox::GraphT& nodes)
{
	size_t n = nodes.size();

	std::vector<size_t> distance;
	types::StrUMapT<size_t> vertices;
	floyd_warshall(distance, vertices, nodes);

	auto dist = sqr_curry(distance, n);

	size_t nsets = 0;
	std::unordered_set<size_t> visited;
	std::unordered_map<size_t,size_t> disjoints;
	for (auto vert : vertices)
	{
		size_t vertex = vert.second;
		if (estd::has(visited, vertex))
		{
			continue;
		}
		visited.emplace(vert.second);
		for (size_t i = 0; i < n; ++i)
		{
			if (dist(vertex, i) < n)
			{
				visited.emplace(i);
				disjoints.emplace(i, nsets);
			}
		}
		++nsets;
	}

	std::vector<distr::ox::GraphT> out(nsets);
	for (auto vert : vertices)
	{
		auto id = vert.first;
		out[disjoints.at(vert.second)].emplace(id, nodes.at(id));
	}
	return out;
}

distr::ox::TopographyT kmeans (
	const types::StringsT& peers,
	const distr::ox::GraphT& nodes,
	KSelectF select)
{
	if (0 == peers.size())
	{
		return {};
	}

	size_t n = nodes.size();
	size_t k = std::min(n, peers.size());

	types::StrUMapT<size_t> vertices;
	std::vector<size_t> distance;
	floyd_warshall(distance, vertices, nodes);

	auto dist = sqr_curry(distance, n);

	auto means = select(k, vertices);
	std::unordered_map<size_t,size_t> clusters;
	for (size_t i = 0; i < k; ++i)
	{
		clusters.emplace(means[i], i);
	}

	bool changed = true;
	for (size_t u = 0; u < max_try_kmeans && changed; ++u)
	{
		// assignment step
		std::unordered_set<size_t> mean_set(means.begin(), means.end());
		for (size_t i = 0; i < n; ++i)
		{
			if (estd::has(mean_set, i))
			{
				continue;
			}
			size_t closest = 0;
			for (size_t j = 1; j < k; ++j)
			{
				if (dist(i, means[closest]) > dist(i, means[j]))
				{
					closest = j;
				}
			}
			clusters.emplace(i, closest);
		}

		// update step
		std::unordered_map<size_t,size_t> best_maxdist;
		std::vector<size_t> next_means(k);
		for (size_t i = 0; i < n; ++i)
		{
			size_t cluster = clusters.at(i);
			size_t maxdist = 0;
			for (size_t j = 0; j < n; ++j)
			{
				if (clusters.at(j) == cluster)
				{
					if (dist(i, j) < n)
					{
						maxdist = std::max(maxdist, dist(i, j));
					}
				}
			}
			if (false == estd::has(best_maxdist, cluster) ||
				maxdist < best_maxdist[cluster])
			{
				best_maxdist[cluster] = maxdist;
				next_means[cluster] = i;
			}
		}

		changed = next_means != means;
		means = next_means;
	}

	distr::ox::NodesT node_bases;
	node_bases.reserve(vertices.size());
	for (auto& vert : vertices)
	{
		auto node = nodes.at(vert.first);
		size_t color_idx = clusters.at(vert.second);
		node->color_ = peers.at(color_idx);
		node_bases.push_back(node);
	}

	distr::ox::TopographyT out;
	std::unordered_set<distr::ox::iTopographicNode*> nonroots;
	for (auto& node : node_bases)
	{
		auto color = node->color_;
		auto& edges = node->edges_;
		for (auto& edge : edges)
		{
			if (edge->color_.size() > 0 && edge->color_ != color)
			{
				out.emplace(edge->get_name(), edge->color_);
			}
			nonroots.emplace(edge.get());
		}
	}
	// record roots
	for (auto& node : node_bases)
	{
		if (false == estd::has(nonroots, node.get()))
		{
			out.emplace(node->get_name(), node->color_);
		}
	}
	return out;
}

}

#endif
