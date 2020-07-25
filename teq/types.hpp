
#include <memory>
#include <type_traits>

// template <typename T>
// concept tensptr_range = ranges::range<T> && ...;

template <typename R>
using RangeValT = typename std::iterator_traits<
	typename R::iterator>::value_type;

template<typename T>
struct is_shared_ptr : std::false_type {};

template<typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

// template<typename T>
// typename std::enable_if<is_shared_ptr<decltype(std::declval<T>().value)>::value,void>::type
