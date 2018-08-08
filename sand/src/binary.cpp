#include "sand/include/binary.hpp"

#ifdef SAND_BINARY_HPP

namespace sand
{

template <>
void rand_normal<float> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	binary<float>(dest, srcs,
	[](const float& a, const float& b) -> float
	{
		std::normal_distribution<float> dist(a, b);
		return dist(get_engine());
	});
}

template <>
void rand_normal<double> (NodeInfo dest, std::vector<NodeInfo>& srcs,
	MetaEncoder::MetaData)
{
	binary<double>(dest, srcs,
	[](const double& a, const double& b) -> double
	{
		std::normal_distribution<double> dist(a, b);
		return dist(get_engine());
	});
}

}

#endif
