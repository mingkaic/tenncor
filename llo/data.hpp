#include "ade/tensor.hpp"

#include "llo/dtype.hpp"

struct GenericData
{
	GenericData (void) = default;

	GenericData (ade::Shape shape, DTYPE dtype);

	GenericData convert_to (DTYPE dtype) const;

	std::shared_ptr<char> data_;

	ade::Shape shape_;
	DTYPE dtype_ = BAD;
};
