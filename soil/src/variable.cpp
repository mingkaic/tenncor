#include "soil/variable.hpp"

#ifdef VARIABLE_HPP

Varptr Variable::get (Shape shape, DTYPE type)
{
	return Varptr(new Variable(shape, type));
}

std::shared_ptr<char> Variable::calculate (void)
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
		return get_one(shape_, type_);
	}
	return get_zero(shape_, type_);
}

Shape Variable::shape (void) const
{
	return shape_;
}

Variable::Variable (Shape shape, DTYPE type) : shape_(shape), type_(type) {}

Varptr::Varptr (Variable* var) : Nodeptr(var) {}

#endif
