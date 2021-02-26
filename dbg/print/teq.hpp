///
/// teq.hpp
/// dbg
///
/// Purpose:
/// Draw an equation graph as an ascii tree
///

#ifndef DBG_TEQ_HPP
#define DBG_TEQ_HPP

#include "internal/teq/teq.hpp"

#include "dbg/print/tree.hpp"

/// Map tensor to label
using LabelsMapT = teq::TensMapT<std::string>;

const std::string dummy_label = "DEPENDENCIES";

struct PrintEqConfig
{
	/// Print every tensor's shape if true, otherwise don't
	bool showshape_ = false;

	bool showtype_ = false;

	bool showvers_ = false;

	/// For every label associated with a tensor, show LABEL=value in the tree
	LabelsMapT labels_;

	bool lsattrs_ = false;

	std::string showattr_;
};

void ten_stream (std::ostream& out,
	teq::iTensor*& root, const std::string& prefix,
	const PrintEqConfig& cfg);

/// Use PrettyTree to render teq::TensptrT graph as an ascii art
struct PrettyEquation final
{
	PrettyEquation (void) : renderer_(
	[](teq::iTensor*& root, size_t) -> teq::TensT
	{
		if (auto f = dynamic_cast<teq::iFunctor*>(root))
		{
			auto args = f->get_args();
			teq::TensT tens;
			tens.reserve(args.size());
			std::transform(args.begin(), args.end(),
				std::back_inserter(tens),
				[](teq::TensptrT child)
				{
					return child.get();
				});
			return tens;
		}
		return {};
	},
	[this](std::ostream& out, teq::iTensor*& root, const std::string& prefix)
	{
		ten_stream(out, root, prefix, this->cfg_);
	}) {}

	/// Stream equation of ptr to out
	void print (std::ostream& out, const teq::TensptrT& ptr)
	{
		renderer_.print(out, ptr.get());
	}

	/// Stream equation of raw ptr to out
	void print (std::ostream& out, teq::iTensor* ptr)
	{
		renderer_.print(out, ptr);
	}

	PrintEqConfig cfg_;

private:
	/// Actual ascii renderer
	PrettyTree<teq::iTensor*> renderer_;
};

#endif // DBG_TEQ_HPP
