---
title: "Practical Memory Safety"
date: 2026-08-01T08:00:00+00:00
tags: ["rust", "zig", "thoughts"]
type: post
showTableOfContents: false
image: "/2026-08-unsafe-water/non_buoyant_s.webp"
weight: 1
build: 
  list: never
---

<style>
.post .post-content {
    margin-top: 0px;
}
</style>


Recently:
1. Zig is [considering][zig-fil-c] a memory safe [Fil-C]-like compilation target.
2. Fil-C is a "memory safe implementation of C", and the Zig issue claims this will be "truly memory safe".
3. This is presented as "unlike Rust", which has escape hatches a.k.a. [Unsafe Rust].
4. But it turns out Fil-C's [definition of memory safety][fil-c-def] is different from Rust's.
5. So maybe Fil-C is also [unsafe](#unsafe-fil-c)?

[zig-unsafe]: {{< ref "/posts/2025-05-zig-mem-safety.md" >}}
[zig-fil-c]: https://codeberg.org/ziglang/zig/issues/36237
[Fil-C]: https://fil-c.org/
[Unsafe Rust]: https://doc.rust-lang.org/book/ch20-01-unsafe-rust.html
[fil-c-def]: https://fil-c.org/#:~:text=Memory%20Safety%3A%20Advanced%20runtime%20checks%20to%20prevent%20exploitable%20memory%20safety%20errors.

You might look at all this and be like _I Ain't Reading All That_. Well, here's the gist.

## Unsafe Fil-C

Abbreviated from [this lobste.rs comment](https://lobste.rs/c/ay4ynr), here's some C code:

```c
struct User {
    char name[8];
    int is_root;
};

struct User* user = malloc(sizeof(struct User));
strcpy(user->name, argv[1]);
```

This code allocates a `User`, and copies a string from the first command-line argument.

But what happens when the input is longer than 8?

```c
if (user->is_root) {
    printf("I am root!\n");
} else {
    printf("I am not root :(\n");
}
```

This can print `I am root!`, even in Fil-C, because `strcpy` will happily overwrite memory that you own.
Fil-C's definition of memory safety is having "runtime checks to prevent exploitable memory safety errors".
The above code is not _technically_ "exploitable" so the overwrite is allowed.

This code _will_ trap if you write _outside the allocation_, which is rounded up from 12 to 16 bytes in this case - 
so technically, the proposed Zig mode _would be safer_ than plain Zig.
See also [steveklabnik's comment on HN](https://news.ycombinator.com/item?id=49087952).

Would you consider this "an exploit"? Would you consider this code "safe"?

## Practical Safety

These are big questions, and though I love a good technically correct answer, 
I’d like to contribute a _less_ nuanced answer to "what is memory safety", which I think is more practical:

**Memory safe code behaves like we intuitively expect it to behave, <u>unless</u> there’s a big sign that says otherwise**.

Which in my head is also the "non-buoyant water law of memory safety".

<figure style="text-align: center;">
  <img
    src="/2026-08-unsafe-water/water_has_no_buoyancy.webp"
    alt="Danger sign warning that the water has no buoyancy"
    width="640"
    height="426"
    style="width: 70%; height: auto;"
  >

  <figcaption>
    Source:
    <a href="https://www.reddit.com/r/thalassophobia/comments/9r921s/theres_something_particularly_terrifying_about/">
      Reddit
    </a>
  </figcaption>

  <figcaption>
    See also:
    <a href="https://www.youtube.com/watch?v=ey06E4iEXzg">
      "Is NON-BUOYANT WATER Deadly?", YouTube
    </a>
  </figcaption>
</figure>

This definition is, obviously, no good for the real experts who design memory models and compilers and CPUs and stuff - they'd like to do ~~fun~~ important stuff like _proving_ properties of programs, and eliminating operations that cannot be observed, and so on. \
So to be more precise: having a proper _technical_ definition of memory safety **is important** - insofar as it ensures that the _safe_ operations on memory match the expectations of users.

The unsafe side of the boundary can have complex and even obscure rules, \
but if the safe side _also_ has obscure rules users must follow, is it _useful to draw the boundary_?

If there's an asterisk on each memory access, then what's the point of "safety" anyway?

As a final analogy, consider control flow.
We do not consider every `if` statement in existence to be "unsafe" just because bad conditions and logical bugs can lead to exploits, memory related or not. It is safe because control flow goes where the source says it can go.
They are "safe" simply _if_ (🥁) they do what they promise to do.

Or, to borrow [Ralf Jung’s phrasing][what-the-hardware-does] slightly out of context: "When writing safe Rust, you do not have to worry" about the abstract machine. And really, the world provides enough things to worry about without adding that to the list.

[what-the-hardware-does]: https://www.ralfj.de/blog/2019/07/14/uninit.html#:~:text=When%20writing%20safe%20Rust%2C%20you%20do%20not%20have%20to%20worry%20about%20this

## Appendix: Some More Examples

### Python CFFI

Python is memory safe: you can use CFFI to do whatever you want:

```python
from cffi import FFI
ffibuilder = FFI()
ffibuilder.cdef("...")

# Or
from ctypes import *
cdll.LoadLibrary("libc.so.6")
libc = CDLL("libc.so.6")
```

But everything is literally called C-something. That's the sign.

Similarly, any language where FFI is an explicit construct (`unsafe extern "C"` in Rust, P/Invoke in .Net, Java's JNI),
_can_ be safe.

### Rust `/proc/self/mem`

A language is _not_ unsafe just because it lets you write to memory like this:

```rust
let mut f = fs::OpenOptions::new()
    .write(true)
    .open("/proc/self/mem").unwrap();
```

Yes you can do really bad memory crimes with this. 
However, there's a clear sign: `/proc/self/mem` is a great way to say "all bets are off". 

### Zig is Unsafe

Not controversial anyway, but see [a previous post]({{< ref "/posts/2025-05-zig-mem-safety.md" >}}).
