
#ifndef EIGEN_MEMORY_HPP
#define EIGEN_MEMORY_HPP

#include <cstdlib>

namespace eigen
{

struct iRuntimeMemory
{
	virtual ~iRuntimeMemory (void) = default;

	virtual void* allocate (size_t size) = 0;

	virtual void deallocate (void* ptr) = 0;
};

// Manage temporary memory
struct RuntimeMemory final : public iRuntimeMemory
{
	void* allocate (size_t size) override
	{
		return malloc(size);
	}

	void deallocate (void* ptr) override
	{
		free(ptr);
	}
};

template <typename T>
struct Expirable final
{
	void expire (void)
	{
		if (ptr_ != nullptr)
		{
			allocator_->deallocate(ptr_);
			ptr_ = nullptr;
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
	void borrow (iRuntimeMemory& memory, size_t nelems, size_t ttl)
	{
		if (false == is_expired())
		{
			global::throw_err("cannot borrow memory when Expirable is not expired");
		}
		ptr_ = (T*) memory.allocate(sizeof(T) * nelems);
		allocator_ = &memory;
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

	iRuntimeMemory* allocator_ = nullptr;
};

}

#endif // EIGEN_MEMORY_HPP
