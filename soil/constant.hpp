#include "sand/node.hpp"

#include "util/error.hpp"

#ifndef SOIL_CONSTANT_HPP
#define SOIL_CONSTANT_HPP

struct Constant final : public Node
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
		return Nodeptr(new Constant((char*) &data[0], shape, get_type<T>()));
	}

	static Nodeptr get (char* data, Shape shape, DTYPE type);

	Constant (const Constant&) = default;

	Constant (Constant&&) = delete;

	Constant& operator = (const Constant&) = default;

	Constant& operator = (Constant&&) = delete;

	std::shared_ptr<char> calculate (Pool& pool) override;

	Nodeptr gradient (Nodeptr& leaf) const override;

private:
	Constant (char* data, Shape shape, DTYPE type);

	std::shared_ptr<char> data_;
};

Nodeptr get_zero (Meta info);

Nodeptr get_one (Meta info);

Nodeptr get_identity (DimT d, DTYPE type, Shape inner = std::vector<DimT>{1});

#endif /* SOIL_CONSTANT_HPP */
