//
//  constant.ipp
//  kiln
//

#ifdef KILN_CONSTANT_HPP

namespace kiln
{

template <typename T>
Constant* Constant::get (T scalar, Graph& graph)
{
	clay::DTYPE dtype = clay::get_type<T>();
	if (clay::DTYPE::BAD == dtype)
	{
		throw clay::UnsupportedTypeError(dtype);
	}
	std::string label = std::string(ioutil::Stream() << scalar);
	std::shared_ptr<char> ptr = clay::make_char(sizeof(T));
	memcpy(ptr.get(), (char*) &scalar, sizeof(T));
	return new Constant(ptr, std::vector<size_t>{1},
		dtype, label, graph);
}

template <typename T>
Constant* Constant::get (std::vector<T> vec, clay::Shape shape, Graph& graph)
{
	clay::DTYPE dtype = clay::get_type<T>();
	if (clay::DTYPE::BAD == dtype)
	{
		throw clay::UnsupportedTypeError(dtype);
	}
	std::string label = (ioutil::Stream() << vec[0] << "...");
	size_t n = vec.size();
	assert(shape.n_elems() == n);
	size_t nbytes = sizeof(T) * n;
	std::shared_ptr<char> ptr = clay::make_char(nbytes);
	memcpy(ptr.get(), (char*) &vec[0], nbytes);
	return new Constant(ptr, shape, dtype,
		label, graph);
}

}

#endif /* KILN_CONSTANT_HPP */
