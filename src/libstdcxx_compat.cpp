/*
 * Compatibility shim for older libstdc++ runtimes that miss
 * std::__throw_bad_array_new_length().
 *
 * Some prebuilt LibTorch CUDA components reference this symbol directly.
 * On systems where libstdc++.so.6 does not export it, dlopen fails at runtime.
 */

extern "C" void __cxa_throw_bad_array_new_length() __attribute__((noreturn));

namespace std
{
[[noreturn]] void __throw_bad_array_new_length()
{
    __cxa_throw_bad_array_new_length();
}
}  // namespace std

