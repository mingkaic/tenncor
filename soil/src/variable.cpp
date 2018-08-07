#include "soil/variable.hpp"

#ifdef SOIL_VARIABLE_HPP

Varptr Variable::get (Shape shape, DTYPE type)
{
	return Varptr(new Variable(shape, type));
}

std::shared_ptr<char> Variable::calculate (Pool& pool)
{
	if (data_ == nullptr)
	{
		handle_error("calculating uninitialized variable");
	}
	return data_;
}

Nodeptr Variable::gradient (Nodeptr& leaf) const
{
	if (this == leaf.get())
	{
		return get_one(info_);
	}
	return get_zero(info_);
}

Variable::Variable (Shape shape, DTYPE type) : Node({shape, type}) {}

Varptr::Varptr (Variable* var) : Nodeptr(var) {}

#endif
