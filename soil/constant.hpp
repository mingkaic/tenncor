#include "soil/inode.hpp"
#include "soil/error.hpp"

#ifndef CONSTANT_HPP
#define CONSTANT_HPP

struct Constant final : public iNode
{
	template <typename T>
	static Nodeptr get (Shape shape, std::vector<T> data)
	{
		if (data.size() != shape.n_elems())
		{
			handle_error("vector size not fitting shape",
				ErrArg<size_t>{"vecsize", data.size()},
				ErrArg<std::string>{"shape", shape.to_string()});
		}
		return Nodeptr(new Constant((char*) &data[0], get_type<T>(), shape));
	}

	static Nodeptr get (char* data, DTYPE type, Shape shape);

	std::shared_ptr<char> calculate (void) override;

	Nodeptr gradient (Nodeptr& leaf) const override;

	Shape shape (void) const override;

	DTYPE type (void) const override
	{
		return type_;
	}

private:
	Constant (char* data, DTYPE type, Shape shape);

	Constant (const Constant&) = default;

	Constant (Constant&&) = delete;

	Constant& operator = (const Constant&) = default;

	Constant& operator = (Constant&&) = delete;

	std::shared_ptr<char> data_;
	Shape shape_;
	DTYPE type_;
};

Nodeptr get_zero (Shape shape, DTYPE type);

Nodeptr get_one (Shape shape, DTYPE type);

Nodeptr get_identity (DimT d, DTYPE type, Shape inner = std::vector<DimT>{1});

#endif /* CONSTANT_HPP */
