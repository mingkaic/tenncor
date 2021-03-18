#include "tenncor/hone/matchain.hpp"
#include "tenncor/hone/cstrules.hpp"

#ifdef HONE_MATCHAIN_HPP

namespace hone
{

template <typename IT>
using ItValT = typename std::iterator_traits<IT>::value_type;

static const std::string lmatmul_pattern = "{"
	"\"op\":{"
		"\"opname\":\"MATMUL\","
		"\"args\":[{"
			"\"op\":{"
				"\"opname\":\"MATMUL\","
				"\"args\":[{"
					"\"symb\":\"X\""
				"},{"
					"\"symb\":\"Y\""
				"}],"
				"\"capture\": \"chained\""
			"}"
		"},{"
			"\"symb\":\"Z\""
		"}]"
	"}"
"}";

static const std::string rmatmul_pattern = "{"
	"\"op\":{"
		"\"opname\":\"MATMUL\","
		"\"args\":[{"
			"\"symb\":\"X\""
		"},{"
			"\"op\":{"
				"\"opname\":\"MATMUL\","
				"\"args\":[{"
					"\"symb\":\"Y\""
				"},{"
					"\"symb\":\"Z\""
				"}],"
				"\"capture\": \"chained\""
			"}"
		"}]"
	"}"
"}";

static bool is_const (teq::iTensor* tens)
{
	if (auto leaf = dynamic_cast<teq::iLeaf*>(tens))
	{
		return leaf->get_usage() == teq::Usage::IMMUTABLE;
	}
	return false;
}

static teq::TensptrsT merge_subsequent_constants (const teq::TensptrsT& chain,
	const global::CfgMapptrT& ctx)
{
	bool prev_cst = is_const(chain.front().get());
	teq::TensptrsT nextchain = {chain.front()};
	for (size_t i = 1, n = chain.size(); i < n; ++i)
	{
		auto is_cst = is_const(chain[i].get());
		if (is_cst && prev_cst)
		{
			// merge
			auto sim = eteq::make_functor(
				egen::MATMUL, {chain[i - 1], chain[i]});
			nextchain.back() = constantize(sim.get(), ctx);
		}
		else
		{
			nextchain.push_back(chain[i]);
		}
		prev_cst = is_cst;
	}
	return nextchain;
}

static std::string debug (const teq::DimsT& dims, size_t* kp, size_t i, size_t j)
{
	if (i == j)
	{
		std::stringstream ss;
		ss << dims[i - 1] << "x" << dims[i];
		return ss.str();
	}
	size_t k = kp[i * dims.size() + j];
	return "(" + debug(dims, kp, i, k) + "@" + debug(dims, kp, k + 1, j) + ")";
}

static inline size_t estimate_ops (
	const numbers::Fraction& ldensity, const numbers::Fraction& rdensity,
	teq::DimT common_dim, teq::DimT lrows, teq::DimT rcols)
{
	return double(ldensity * rdensity) * common_dim * lrows * rcols;
}

// Given kp = size_t[n * n] where n = dims.size(), populate kp st
// kp[i * n + j] -> k where k denotes the index to optimally place matmul
// arr[:k] @ arr[k:]
// dims are the row x col or args with repeated subsequent col, row deduplicated
static void optimal_matchain (size_t* kp, const teq::DimsT& dims,
	const std::vector<numbers::Fraction>& density)
{
	size_t n = dims.size();
	teq::DimT dp[n][n];
	numbers::Fraction densep[n-1][n-1];
	std::memset(dp, 0, sizeof(dp));
	std::memset(kp, 0, sizeof(size_t) * n * n);
	for (size_t i = 0, ndens = density.size(); i < ndens; ++i)
	{
		densep[i][i] = density[i];
	}

	for (size_t l = 2; l < n; l++)
	{
		for (size_t i = 1; i < n - l + 1; i++)
		{
			size_t j = i + l - 1;
			dp[i][j] = std::numeric_limits<teq::DimT>::max();
			for (size_t k = i; k <= j - 1; k++)
			{
				// density = 1 - sparsity, sparsity the probability of any
				// element being zero,
				// new sparsity = P(left row empty) * P(right row empty)
				// = (left sparsity * right sparsity) ^ dims[k]
				const numbers::Fraction& ldense = densep[i-1][k-1];
				const numbers::Fraction& rdense = densep[k][j-1];
				size_t ops = estimate_ops(
					ldense, rdense, dims[k], dims[i-1], dims[j]);
				size_t q = dp[i][k] + dp[k + 1][j] + ops;
				if (q < dp[i][j])
				{
					dp[i][j] = q;
					kp[i * n + j] = k;
					densep[i-1][j-1] = eigen::matmul_density(
						ldense, rdense, dims[k]);
				}
			}
		}
	}
}

static teq::TensptrT rechain (size_t* kp, const teq::TensptrsT& args,
	size_t i, size_t j)
{
	if (i == j)
	{
		return args[i - 1];
	}
	size_t k = kp[i * (args.size() + 1) + j];
	return eteq::make_functor(egen::MATMUL, {
		rechain(kp, args, i, k),
		rechain(kp, args, k + 1, j)
	});
}

// Using an optimization info graph, find subtrees comprised solely of
// contiguous matrix multiplication operations and flatten the multiplication
// arguments to a "chain" and map it to the subtree root.
// (special case:) In cases when a single matmul node is reused by
// different subtrees, clip the reused node from both parent subtrees
// (treating the reuse node as an argument for its parents and itself as a
// root of a new subtree).
void flatten_matmul_hierarchy (
	types::PairsT<teq::iTensor*,teq::TensptrsT>& chain_roots,
	const opt::GraphInfo& graph)
{
	// look for matrix chains (subtrees comprised of subsequent MATMUL functors)
	query::Node lcond;
	query::Node rcond;
	{
		std::stringstream ss;
		ss << lmatmul_pattern;
		query::json_parse(lcond, ss);
	}
	{
		std::stringstream ss;
		ss << rmatmul_pattern;
		query::json_parse(rcond, ss);
	}
	teq::GraphStat stat;
	for (teq::TensptrT root : graph.roots_)
	{
		root->accept(stat);
	}
	auto orderfunc = [&](teq::iTensor* a, teq::iTensor* b)
	{
		return stat.graphsize_[a].upper_ < stat.graphsize_[b].upper_;
	};
	query::QResultsT lmatmul = graph.sindex_.match(lcond);
	query::QResultsT rmatmul = graph.sindex_.match(rcond);
	teq::TensMapT<teq::TensT> parents;
	teq::TensSetT args;
	teq::TensSetT shareds;
	for (auto& result : lmatmul)
	{
		auto parent = result.root_;
		auto child = result.symbs_.at("chained");
		if (estd::has(args, child))
		{
			shareds.emplace(child);
		}
		args.emplace(child);
		parents.emplace(parent, teq::TensT{child, nullptr});
	}
	for (auto& result : rmatmul)
	{
		auto parent = result.root_;
		auto child = result.symbs_.at("chained");
		if (estd::has(args, child))
		{
			shareds.emplace(child);
		}
		args.emplace(child);
		if (estd::has(parents, parent))
		{
			parents[parent][1] = child;
		}
		else
		{
			parents.emplace(parent, teq::TensT{nullptr, child});
		}
	}

	// roots can ignore the matmuls without matmul arguments
	teq::TensSetT roots = estd::map_keyset(parents);
	std::list<teq::iTensor*> orders(roots.begin(), roots.end());
	orders.sort(orderfunc);

	teq::TensMapT<teq::TensptrsT> chains;
	teq::TensMapT<std::string> orig_order;
	teq::TensMapT<size_t> orig_nops;
	for (auto order : orders)
	{
		teq::TensptrsT chain;
		auto parent = static_cast<teq::iFunctor*>(order);
		auto children = parent->get_args();
		auto& refs = parents.at(order);
		orig_nops[order] = children[0]->shape().n_elems() *
			children[1]->shape().at(1);
		for (size_t i = 0, n = refs.size(); i < n; ++i)
		{
			if (nullptr == refs[i] || estd::has(shareds, refs[i]))
			{
				chain.push_back(children[i]);
				orig_order[order] += children[i]->shape().to_string();
			}
			else
			{
				auto subref = static_cast<teq::iFunctor*>(refs[i]);
				if (estd::has(parents, subref))
				{
					auto& subchain = chains.at(subref);
					chain.insert(chain.end(), subchain.begin(), subchain.end());
					orig_order[order] += "(" + orig_order[subref] + ")";
					roots.erase(subref);
					orig_nops[order] += orig_nops[subref];
				}
				else
				{
					auto subchildren = subref->get_args();
					chain.insert(chain.end(),
						subchildren.begin(), subchildren.end());
					orig_order[order] += "(" +
						subchildren[0]->shape().to_string() +
						subchildren[1]->shape().to_string() + ")";
					orig_nops[order] +=
						subchildren[0]->shape().n_elems() *
						subchildren[1]->shape().at(1);
				}
			}
		}
		chains.emplace(parent, chain);
	}
	roots.insert(shareds.begin(), shareds.end());
	std::list<teq::iTensor*> ordroots(roots.begin(), roots.end());
	ordroots.sort(orderfunc);
	for (auto root : ordroots)
	{
		if (estd::has(chains, root))
		{
			chain_roots.push_back({root, chains.at(root)});
		}
	}
}

void matrix_chain (opt::GraphInfo& graph)
{
	types::PairsT<teq::iTensor*,teq::TensptrsT> chains;
	flatten_matmul_hierarchy(chains, graph);
	teq::OwnMapT converts;
	for (auto& rootchain : chains)
	{
		auto& root = rootchain.first;
		auto& chain = rootchain.second;
		if (chain.size() <= 2)
		{
			continue;
		}
		chain = merge_subsequent_constants(chain, global::context());
		for (size_t i = 0, n = chain.size(); i < n; ++i)
		{
			auto link = chain[i];
			if (estd::has(converts, link.get()))
			{
				chain[i] = converts.at(link.get());
			}
		}

		teq::DimsT dims = {chain.front()->shape().at(1)};
		std::vector<numbers::Fraction> density;
		dims.reserve(chain.size() + 1);
		density.reserve(chain.size());

		eigen::Device device;
		auto ctx = global::context();
		teq::TensSetT chainraws;
		teq::multi_get(chain.begin(), chain.end(),
			std::inserter(chainraws, chainraws.end()));
		teq::get_eval(ctx).evaluate(device, chainraws);
		for (auto link : chain)
		{
			dims.push_back(link->shape().at(0));
			density.push_back(eigen::calc_density(*link));
		}
		size_t n = dims.size();
		size_t kp[n * n];
		optimal_matchain(kp, dims, density);
		auto replacement = rechain(kp, chain, 1, n - 1);
		converts.emplace(root, replacement);
	}
	graph.replace(converts);
}

}

#endif
