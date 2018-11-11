#include "age/files.hpp"

#ifndef AGE_PARSER_HPP
#define AGE_PARSER_HPP

void unmarshal_json (File& runtime_file, File& api_file,
    std::istream& jstr);

#endif // AGE_PARSER_HPP
