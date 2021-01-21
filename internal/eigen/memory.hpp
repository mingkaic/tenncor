
#ifndef EIGEN_MEMORY_HPP
#define EIGEN_MEMORY_HPP

#include <cstdlib>

#include <boost/pool/pool.hpp>

#include "internal/global/global.hpp"

namespace eigen
{

struct iRuntimeMemory
{
	virtual ~iRuntimeMemory (void) = default;

	virtual void* allocate (size_t size) = 0;

	virtual void deallocate (void* ptr, size_t size) = 0;
};

using RTMemptrT = std::shared_ptr<iRuntimeMemory>;

// Manage temporary memory
struct RuntimeMemory final : public iRuntimeMemory
{
	void* allocate (size_t size) override
	{
		return malloc(size);
	}

	void deallocate (void* ptr, size_t) override
	{
		free(ptr);
	}
};

// Manage temporary memory using boost pool
struct BoostRuntimeMemory final : public iRuntimeMemory
{
	BoostRuntimeMemory (void) : pool_(sizeof(char)) {}

	void* allocate (size_t size) override
	{
		return pool_.ordered_malloc(size);
	}

	void deallocate (void* ptr, size_t size) override
	{
		pool_.ordered_free(ptr, size);
	}

	boost::pool<> pool_;
};

template <typename T>
struct Expirable final
{
	void expire (void)
	{
		if (ptr_ != nullptr)
		{
			allocator_->deallocate(ptr_, size_);
			ptr_ = nullptr;
			size_ = 0;
			allocator_ = nullptr;
		}
		ttl_ = 0;
	}

	void tick (void)
	{
		if (0 == (--ttl_))
		{
			expire();
		}
	}

	bool is_expired (void) const
	{
		return 0 == ttl_;
	}

	size_t get_ttl (void) const
	{
		return ttl_;
	}

	T* get (void)
	{
		return ptr_;
	}

	std::string debug_info (void)
	{
		return fmts::sprintf("runtime[%p],ttl=%d", ptr_, ttl_);
	}

	// borrow memory from runtime memory
	void borrow (RTMemptrT& memory, size_t nelems, size_t ttl)
	{
		if (nullptr == memory)
		{
			global::fatal("cannot borrow from null memory");
		}
		if (false == is_expired())
		{
			global::throw_err("cannot borrow memory when Expirable is not expired");
		}
		size_ = sizeof(T) * nelems;
		ptr_ = (T*) memory->allocate(size_);
		allocator_ = memory;
		extend_life(ttl);
	}

	void extend_life (size_t ttl)
	{
		if (nullptr == ptr_)
		{
			global::fatal("cannot extend ttl of expired Expirable");
		}
		if (ttl_ < ttl)
		{
			ttl_ = ttl;
		}
	}

private:
	friend struct iRuntimeMemory;

	size_t ttl_ = 0;

	T* ptr_ = nullptr;

	size_t size_ = 0;

	RTMemptrT allocator_ = nullptr;
};

void set_runtime (RTMemptrT mem, global::CfgMapptrT ctx = global::context());

RTMemptrT get_runtime (const global::CfgMapptrT& ctx = global::context());

}

#endif // EIGEN_MEMORY_HPP
