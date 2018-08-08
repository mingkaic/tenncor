#include "sand/include/unary.hpp"

#ifdef SAND_UNARY_HPP

namespace sand
{

template <>
void abs<uint8_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	Shape srcshape = srcs[0].shape_;
	uint8_t* destdata = (uint8_t*) dest.data_;
	uint8_t* srcdata = (uint8_t*) srcs[0].data_;
	std::memcpy(destdata, srcdata, sizeof(uint8_t) * srcshape.n_elems());
}

template <>
void abs<uint16_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	Shape srcshape = srcs[0].shape_;
	uint16_t* destdata = (uint16_t*) dest.data_;
	uint16_t* srcdata = (uint16_t*) srcs[0].data_;
	std::memcpy(destdata, srcdata, sizeof(uint16_t) * srcshape.n_elems());
}

template <>
void abs<uint32_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	Shape srcshape = srcs[0].shape_;
	uint32_t* destdata = (uint32_t*) dest.data_;
	uint32_t* srcdata = (uint32_t*) srcs[0].data_;
	std::memcpy(destdata, srcdata, sizeof(uint32_t) * srcshape.n_elems());
}

template <>
void abs<uint64_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	Shape srcshape = srcs[0].shape_;
	uint64_t* destdata = (uint64_t*) dest.data_;
	uint64_t* srcdata = (uint64_t*) srcs[0].data_;
	std::memcpy(destdata, srcdata, sizeof(uint64_t) * srcshape.n_elems());
}

template <>
void neg<uint8_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	throw std::bad_function_call();
}

template <>
void neg<uint16_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	throw std::bad_function_call();
}

template <>
void neg<uint32_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	throw std::bad_function_call();
}

template <>
void neg<uint64_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	throw std::bad_function_call();
}

template <>
void n_elems<uint32_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	if (1 != srcs.size())
	{
		handle_error("transpose requires 1 arguments",
			ErrArg<size_t>{"num_args", srcs.size()});
	}

	uint32_t* destdata = (uint32_t*) dest.data_;
	destdata[0] = srcs[0].shape_.n_elems();
}

template <>
void n_dims<uint8_t> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData mdata)
{
	if (1 != srcs.size())
	{
		handle_error("transpose requires 1 arguments",
			ErrArg<size_t>{"num_args", srcs.size()});
	}

	uint8_t* destdata = (uint8_t*) dest.data_;
	if (mdata[0] > 0)
	{
		destdata[0] = srcs[0].shape_.at(mdata[0] - 1);
	}
	else
	{
		destdata[0] = srcs[0].shape_.n_elems();
	}
}

}

#endif
