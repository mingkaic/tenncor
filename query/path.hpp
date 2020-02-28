#ifndef QUERY_PATH_HPP
#define QUERY_PATH_HPP

#include <memory>
#include <string>
#include <vector>

#include "estd/contain.hpp"

#include "teq/teq.hpp"

namespace query
{

using MemoryT = std::unordered_set<size_t>;

using SymbMapT = std::unordered_map<std::string,teq::iTensor*>;

template <typename T>
using EnumsT = std::vector<std::pair<size_t,T>>;

struct Path;

using PathptrT = std::shared_ptr<Path>;

using PrevT = std::pair<size_t,PathptrT>;

struct Path final
{
	Path (teq::iTensor* node,
		PrevT prev = {0, nullptr},
		SymbMapT symbols = {},
		MemoryT memory = {}) :
		tens_(node), symbols_(symbols),
		prev_(prev), memory_(memory)
	{
		assert(nullptr != tens_);
	}

	Path (const Path& other,
		SymbMapT symbols = {},
		MemoryT memory = {}) :
		tens_(other.tens_), symbols_(symbols),
		prev_(other.prev_), memory_(memory)
	{
		assert(nullptr != tens_);
	}

	Path (const Path& other) = default;
	Path (Path&& other) = default;
	Path& operator = (const Path& other) = default;
	Path& operator = (Path&& other) = default;

	EnumsT<teq::iTensor*> get_args (void) const
	{
		auto f = dynamic_cast<teq::iFunctor*>(tens_);
		EnumsT<teq::iTensor*> out;
		if (nullptr == f)
		{
			return out;
		}
		auto children = f->get_children();
		size_t nargs = children.size();
		out.reserve(nargs - memory_.size());
		for (size_t i = 0; i < nargs; ++i)
		{
			if (false == estd::has(memory_, i))
			{
				out.push_back(std::pair<size_t,teq::iTensor*>{i, children[i].get()});
			}
		}
		return out;
	}

	PathptrT recall (void)
	{
		auto last = prev_.second;
		if (nullptr == last)
		{
			return nullptr;
		}
		MemoryT next_mem = last->memory_;
		next_mem.emplace(prev_.first);
		return std::make_shared<Path>(*last, symbols_, next_mem);
	}

	teq::iTensor* tens_;

	SymbMapT symbols_;

	PrevT prev_;

	MemoryT memory_;
};

using PathsT = std::vector<PathptrT>;

}

#endif // QUERY_PATH_HPP
