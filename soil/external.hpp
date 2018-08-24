#include "soil/functor.hpp"

#include "util/sorted_arr.hpp"

using Range = SortedArr<uint8_t,2>;

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

template <typename T>
Nodeptr cast (Nodeptr a)
{
	return cast(a, get_type<T>());
}

Nodeptr cast (Nodeptr a, DTYPE type);

Nodeptr abs (Nodeptr a);

Nodeptr operator - (Nodeptr a);

Nodeptr operator ! (Nodeptr a);

Nodeptr sin (Nodeptr a);

Nodeptr cos (Nodeptr a);

Nodeptr tan (Nodeptr a);

Nodeptr exp (Nodeptr a);

Nodeptr log (Nodeptr a);

Nodeptr sqrt (Nodeptr a);

Nodeptr round (Nodeptr a);

//! flip a values in specified dimensions
Nodeptr flip (Nodeptr a, uint8_t dim);

//! transpose dimensions or groups of consecutive dimensions,
//! default is first 2 dimensions
Nodeptr transpose (Nodeptr a);

Nodeptr transpose (Nodeptr a, std::pair<Range,Range> groups);

Nodeptr transpose (Nodeptr a, std::pair<uint8_t,uint8_t> dims);

//! get the number of elements in a
Nodeptr n_elems (Nodeptr a);

//! get the number of elements in across a dimension in a
Nodeptr n_dims (Nodeptr a);

Nodeptr n_dims (Nodeptr a, uint8_t d);

//! groups everything
Nodeptr group (Nodeptr a);

//! group dimensions in group range
Nodeptr group (Nodeptr a, Range range);


//! b to the power of x
Nodeptr pow (Nodeptr base, Nodeptr xponent);

Nodeptr operator + (Nodeptr a, Nodeptr b);

Nodeptr operator - (Nodeptr a, Nodeptr b);

Nodeptr operator * (Nodeptr a, Nodeptr b);

Nodeptr operator / (Nodeptr a, Nodeptr b);

Nodeptr operator == (Nodeptr a, Nodeptr b);

Nodeptr operator != (Nodeptr a, Nodeptr b);

Nodeptr operator < (Nodeptr a, Nodeptr b);

Nodeptr operator > (Nodeptr a, Nodeptr b);

//! matrix multiplication (todo: expand to include matmul along other dimensions, currently {0, 1} only)
Nodeptr matmul (Nodeptr a, Nodeptr b);

//! generate data of within binomial distribution given (n, p)
Nodeptr binomial_sample (Nodeptr n, Nodeptr p);

//! generate data of within uniform distribution given (min, max)
Nodeptr uniform_sample (Nodeptr a, Nodeptr b);

//! generate data of within normal distribution given (mean, stdev)
Nodeptr normal_sample (Nodeptr mean, Nodeptr stdev);


//! obtain the index of max value in a, lack of or invalid dimension look across all of a
Nodeptr arg_max (Nodeptr a);

Nodeptr arg_max (Nodeptr a, uint64_t dim);

//! obtain the max of a, lack of or invalid dimension look across all of a
Nodeptr reduce_max (Nodeptr a);

Nodeptr reduce_max (Nodeptr a, uint64_t dim);

//! obtain the sum of a, lack of or invalid dimension look across all of a
Nodeptr reduce_sum (Nodeptr a);

Nodeptr reduce_sum (Nodeptr a, uint64_t dim);
