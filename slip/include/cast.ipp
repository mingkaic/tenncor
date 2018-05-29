//
//  cast.ipp
//  slip
//

#ifdef SLIP_CAST_HPP

namespace slip
{

template <typename T>
void cast (clay::State& dest, std::vector<clay::State> srcs)
{
	clay::Shape& shape = dest.shape_;
	size_t n = shape.n_elems();
	T* d = safe_get<T>(dest.data_);
	const char* s = safe_get<const char>(srcs.back().data_);
	clay::DTYPE srctype = srcs.back().dtype_;
	if (srctype == dest.dtype_)
	{
		std::memcpy(d, s, n * sizeof(T));
		return;
	}
	switch (srctype)
	{
		case clay::DOUBLE:
		{
			const double* ptr = (const double*) s;
			std::transform(ptr, ptr + n, d,
			[](double d) -> T
			{
				return d;
			});
		}
		break;
		case clay::FLOAT:
		{
			const float* ptr = (const float*) s;
			std::transform(ptr, ptr + n, d,
			[](float d) -> T
			{
				return d;
			});
		}
		break;
		case clay::INT8:
		{
			const int8_t* ptr = (const int8_t*) s;
			std::transform(ptr, ptr + n, d,
			[](int8_t d) -> T
			{
				return d;
			});
		}
		break;
		case clay::UINT8:
		{
			const uint8_t* ptr = (const uint8_t*) s;
			std::transform(ptr, ptr + n, d,
			[](uint8_t d) -> T
			{
				return d;
			});
		}
		break;
		case clay::INT16:
		{
			const int16_t* ptr = (const int16_t*) s;
			std::transform(ptr, ptr + n, d,
			[](int16_t d) -> T
			{
				return d;
			});
		}
		break;
		case clay::UINT16:
		{
			const uint16_t* ptr = (const uint16_t*) s;
			std::transform(ptr, ptr + n, d,
			[](uint16_t d) -> T
			{
				return d;
			});
		}
		break;
		case clay::INT32:
		{
			const int32_t* ptr = (const int32_t*) s;
			std::transform(ptr, ptr + n, d,
			[](int32_t d) -> T
			{
				return d;
			});
		}
		break;
		case clay::UINT32:
		{
			const uint32_t* ptr = (const uint32_t*) s;
			std::transform(ptr, ptr + n, d,
			[](uint32_t d) -> T
			{
				return d;
			});
		}
		break;
		case clay::INT64:
		{
			const int64_t* ptr = (const int64_t*) s;
			std::transform(ptr, ptr + n, d,
			[](int64_t d) -> T
			{
				return d;
			});
		}
		break;
		case clay::UINT64:
		{
			const uint64_t* ptr = (const uint64_t*) s;
			std::transform(ptr, ptr + n, d,
			[](uint64_t d) -> T
			{
				return d;
			});
		}
		break;
		default:
			throw std::exception(); // todo: add context (unsupported type)
	}
}

}

#endif
