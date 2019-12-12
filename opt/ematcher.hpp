#include <unordered_map>

#include "marsh/objs.hpp"

#include "teq/ileaf.hpp"
#include "teq/ifunctor.hpp"

extern "C" {
#include "opt/parse/def.h"
}

#ifndef OPT_EMATCHER_HPP
#define OPT_EMATCHER_HPP

namespace opt
{

struct Candidate final
{
	Candidate (void) = default;

	Candidate (std::string any, teq::TensptrT tens) :
		anys_({{any, tens}}) {}

	/// Map any symbol to associated Tensor
	std::unordered_map<std::string,teq::TensptrT> anys_;

	/// Map variadic symbol to associated Edges
	std::unordered_map<std::string,teq::TensptrsT> variadic_;
};

/// All candidates generated by a matcher
using CandsT = std::vector<Candidate>;

/// Matches matcher function id (fid) to all candidates
using MatchMapT = std::unordered_map<std::string,CandsT>;

/// Mapping of all visited tensors to their matched candidates
using MatchCtxT = std::unordered_map<teq::iTensor*,MatchMapT>;

struct iEdgeMatcher
{
	virtual ~iEdgeMatcher (void) = default;

	/// Return no candidates if matching child doesn't exist or 
	/// have no candidates, otherwise return candidates
	virtual CandsT match (const MatchCtxT& cands, const teq::TensptrT& child) = 0;
};

using EMatchptrT = std::unique_ptr<iEdgeMatcher>;

using EMatchptrsT = std::vector<EMatchptrT>;

struct ScalarEMatcher final : public iEdgeMatcher
{
	ScalarEMatcher (double scalar) : scalar_(scalar) {}

	CandsT match (const MatchCtxT& ctx, const teq::TensptrT& child) override
	{
		CandsT out;
		auto leaf = dynamic_cast<teq::iLeaf*>(child.get());
		if (nullptr != leaf && leaf->is_const())
		{
			teq::Shape shape = child->shape();
			std::vector<double> d(shape.n_elems(), scalar_);
			if (leaf->to_string() == teq::const_encode(d.data(), shape))
			{
				out.push_back(Candidate());
			}
		}
		return out;
	}

private:
	double scalar_;
};

struct AnyEMatcher final : public iEdgeMatcher
{
	AnyEMatcher (std::string any) : any_(any) {}

	CandsT match (const MatchCtxT& ctx, const teq::TensptrT& child) override
	{
		return CandsT{Candidate(any_, child)};
	}

private:
	std::string any_;
};

struct FuncEMatcher final : public iEdgeMatcher
{
	FuncEMatcher (std::string fid) : fid_(fid) {}

	CandsT match (const MatchCtxT& ctx, const teq::TensptrT& child) override
	{
		if (false == estd::has(ctx, child.get()))
		{
			return CandsT{};
		}
		const MatchMapT& m2cands = ctx.at(child.get());
		CandsT out;
		return estd::try_get(m2cands, fid_, out);
	}

private:
	std::string fid_;
};

}

#endif // OPT_EMATCHER_HPP
