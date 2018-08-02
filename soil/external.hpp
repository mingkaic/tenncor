#include "soil/functor.hpp"

struct DataBucket
{
	DataBucket (std::shared_ptr<char> data, DTYPE type, Shape shape) :
		data_(data), shape_(shape), type_(type) {}

	template <typename T>
	std::vector<T> vectorize (void) const
	{
		assert(get_type<T>() == type_);
		T* data = (T*) data_.get();
		return std::vector<T>(data, data + shape_.n_elems());
	}

	char* data (void) const
	{
		return data_.get();
	}

	Shape shape (void) const
	{
		return shape_;
	}

	DTYPE type (void) const
	{
		return type_;
	}

private:
	std::shared_ptr<char> data_;
	Shape shape_;
	DTYPE type_;
};

DataBucket evaluate (Nodeptr& exit_node);

CoordOp dim_swap (std::pair<uint8_t,uint8_t> dims);

Nodeptr group (Nodeptr a);

Nodeptr transpose (Nodeptr a);

Nodeptr transpose (Nodeptr a, CoordOp dim_op);

Nodeptr operator + (Nodeptr a, Nodeptr b);

Nodeptr operator * (Nodeptr a, Nodeptr b);

Nodeptr matmul (Nodeptr a, Nodeptr b);
