#include "soil/functor.hpp"

struct DataBucket : public DataSource
{
	DataBucket (char* data, DTYPE type, Shape shape) :
		DataSource(data, type, shape.n_elems()), shape_(shape) {}

	template <typename T>
	std::vector<T> vectorize (void) const
	{
		assert(get_type<T>() == type_);
		T* data = (T*) data_.get();
		return std::vector<T>(data, data + shape_.n_elems());
	}

	Shape shape (void) const
	{
		return shape_;
	}

private:
	Shape shape_;
};

DataBucket evaluate (Nodeptr& exit_node);
