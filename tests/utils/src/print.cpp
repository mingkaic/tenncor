#include "tensor/tensorshape.hpp"

#include "print.hpp"

#ifdef TTEST_PRINT_HPP

namespace testutils
{

std::string sprintf (const char* fmt...)
{
	va_list args;
	va_start(args, fmt);
	std::stringstream ss;

	while (*fmt != '\0')
	{
		if (*fmt == '%')
		{
			switch (*(++fmt))
			{
				case '%':
					ss << '%';
				break;
				case 'd':
				case 'i':
					ss << va_arg(args, signed);
				break;
				case 'u':
					ss << va_arg(args, size_t);
				break;
				case 'f':
				case 'F':
					ss << va_arg(args, double);
				break;
				case 'c':
					ss << static_cast<char>(va_arg(args, int));
				break;
				case 's':
					ss << va_arg(args, const char*);
				break;
				case 'e': // pointer
					ss << va_arg(args, void*);
				break;
				case 'p': // shape
				{
					nnet::tensorshape* shape = va_arg(args, nnet::tensorshape*);
					print_shape(*shape, ss);
				}
				break;
				case 'v': // vector
				{
					switch (*(++fmt))
					{
						case 'u':
						{
							std::vector<size_t>* v = va_arg(args, std::vector<size_t>*);
							print(*v, ss);
						}
						break;
						case 'f':
						case 'F':
						{
							std::vector<double>* v = va_arg(args, std::vector<double>*);
							print(*v, ss);
						}
						break;
						case 'd':
						case 'i':
						default:
						{
							std::vector<signed>* v = va_arg(args, std::vector<signed>*);
							print(*v, ss);
						}
					}
				}
				break;
			}
			if (*fmt != '\0')
			{
				++fmt;
			}
		}
		else
		{
			ss << *fmt;
			++fmt;
		}
	}
	va_end(args);
	return ss.str();
}


}

#endif
