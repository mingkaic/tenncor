#include <fstream>
//#include <experimental/filesystem>

#include "internal/eigen/eigen.hpp"

#ifndef DBG_PROFILE_NODEDISPLAY_HPP
#define DBG_PROFILE_NODEDISPLAY_HPP

namespace dbg
{

namespace profile
{

//using fs = std::experimental::filesystem;

struct path
{
	path (const std::string& path) : path_(path) {}

	path concat (const std::string& next)
	{
		return path(path_ + "/" + next);
	}

	operator std::string()
	{
		return path_;
	}

	std::string path_;
};

template <typename T>
static void display (std::ostream& os, const teq::iTensor& tens)
{
	if (eigen::is_sparse(tens))
	{
		os << eigen::make_smatmap_ro<T>(tens) << std::endl;
	}
	else
	{
		os << eigen::make_matmap_ro<T>(tens) << std::endl;
	}
}

#define _TYPE_DISPLAY(REAL_TYPE) display<REAL_TYPE>(of, *tens);

struct NodeDisplay final : public teq::iOnceTraveler
{
	NodeDisplay (const teq::TensMapT<std::string>& ids,
		const std::string& outpath = "/tmp") :
		outpath_(outpath), ids_(&ids) {}

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (teq::iLeaf& leaf) override
	{
		display_tens(&leaf);
	}

	/// Implementation of iOnceTraveler
	void visit_func (teq::iFunctor& func) override
	{
		display_tens(&func);
	}

	void display_tens (teq::iTensor* tens)
	{
		std::string uuid;
		if (estd::get(uuid, *ids_, tens))
		{
			std::string outf(outpath_.concat(uuid));
			std::ofstream of(outf.c_str());
			auto dtype = (egen::_GENERATED_DTYPE) tens->get_meta().type_code();
			TYPE_LOOKUP(_TYPE_DISPLAY, dtype);
		}
	}

private:
	//fs::path outpath_;
	path outpath_;

	const teq::TensMapT<std::string>* ids_;
};

#undef _TYPE_DISPLAY

}

}

#endif // DBG_PROFILE_NODEDISPLAY_HPP
