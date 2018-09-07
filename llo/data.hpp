#include "ade/tensor.hpp"

#include "llo/dtype.hpp"

struct iSource
{
	virtual ~iSource (void) = default;

	virtual DTYPE get_type (void) const = 0;
};

struct TptrHasher
{
	size_t operator() (const ade::Tensorptr& p) const
	{
		return (size_t) p.get();
	}
};

struct Session
{
	std::unordered_map<ade::Tensorptr,iSource*,TptrHasher> sources_;
};

template <typename T>
struct Source : public iSource
{
	Source (ade::Shape shape) : data_(shape.n_elems()) {}

	DTYPE get_type (void) const override
	{
		return get_type<T>();
	}

	std::vector<T> data_;
};
