---
title: "Practical Memory Safety"
date: 2026-08-03T04:30:00+00:00
tags: ["rust", "zig", "thoughts"]
type: post
showTableOfContents: false
image: "/2026-08-unsafe-water/non_buoyant_s.webp"
weight: 1
---

<style>
.post .post-content {
    margin-top: 0px;
}
</style>


Recently:
1. Zig is [considering][zig-fil-c] a safe, [Fil-C]-like compilation target.
2. Fil-C is a "memory safe implementation of C", and the Zig issue claims this will be "truly memory safe".
3. This is presented as "unlike Rust" because Rust has escape hatches a.k.a. [Unsafe Rust].
4. But it turns out Fil-C's [definition of memory safety][fil-c-def] is different from Rust's.
5. So maybe Fil-C is also [unsafe](#unsafe-fil-c)?

[zig-unsafe]: {{< ref "/posts/2025-05-zig-mem-safety.md" >}}
[zig-fil-c]: https://codeberg.org/ziglang/zig/issues/36237
[Fil-C]: https://fil-c.org/
[Unsafe Rust]: https://doc.rust-lang.org/book/ch20-01-unsafe-rust.html
[fil-c-def]: https://fil-c.org/#:~:text=Memory%20Safety%3A%20Advanced%20runtime%20checks%20to%20prevent%20exploitable%20memory%20safety%20errors.

You might look at all this and be like, _I Ain't Reading All That_.
And you might get the _vibe_ that memory safety is something fuzzy and controversial[^1],
with no right or wrong answers... so maybe you should care _less_ about it?

But 'tis not so! Not at all!
By the end of this post, I hope to convince you of two seemingly contradictory ideas:

1. Memory safety _can_ be viewed as a spectrum. We can have _more_ of it - which is good even if the details can be nuanced and frustrating.
2. There can be a _practical_ definition of memory safety that _isn't_ about nuance at all - which is also good, especially **for you**.

So, let's dig in, starting with the gist of all of this recent hoo-ha ☝️

[^1]: You might feel that regardless, which is probably more to do with how social media and the internet at large amplify polarization. The good news about our lord and savior Ferris the Crab _can_ sound preachy at times.

## Unsafe Fil-C

Adapted from [this lobste.rs comment](https://lobste.rs/c/ay4ynr), here's some C code:

```c
struct User {
    char name[8];
    int is_root;
};

struct User* user = malloc(sizeof(struct User));
strcpy(user->name, argv[1]);
```

This code allocates a `User` and copies a string from the first command-line argument.

But what happens when the input is 8 characters or longer?

```c
if (user->is_root) {
    printf("I am root!\n");
} else {
    printf("I am not root :(\n");
}
```

This can print `I am root!`, even in Fil-C, because `strcpy` will happily overwrite memory that you own.
Fil-C defines memory safety as having having "runtime checks to prevent exploitable memory safety errors".
The above code is not _technically_ "exploitable" so the overwrite is allowed.

This code _will_ trap if you write _outside the allocation_[^3],
so the proposed Zig mode **would totally be safer than Yolo-Zig**[^2], which is a good thing.
See also [steveklabnik's comment on HN](https://news.ycombinator.com/item?id=49087952).

[^3]: The allocation is rounded up from 12 to 16 bytes in this case.

Clearly, if Fil-C makes this code _more_ memory safe, then memory safety as a whole must be a spectrum!

But would you consider this "an exploit"? Would you consider this code "safe"?

[^2]: Fil-C [calls] classic C "Yolo-C", and since Zig is adopting these semantics, this seems only fair.

[calls]: https://fil-c.org/runtime

## Practical Safety

These are big questions, and though I love a good technically correct answer, 
I’d like to contribute a _less_ nuanced answer to the question "what is memory safety" that I think is more practical:

Memory-safe code is **code where _memory_ has (1) intuitive and (2) easy to follow rules,** \
**<u>unless</u> there’s a big sign that says otherwise**.

In my head, this is also the "non-buoyant water law of memory safety".

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

This definition is, obviously, no good for the real experts who design memory models and compilers and CPUs - they'd like to do ~~fun~~ important stuff like _proving_ properties of programs, and eliminating operations that cannot be observed, and so on. \

So to be more precise: having a proper _technical_ definition of memory safety **is important** - insofar as it ensures that _safe_ operations on memory match our general expectations around memory and have rules that can easily be followed correctly.

The unsafe side of the boundary can have complex and even obscure rules, \
but if the safe side _also_ has obscure rules that are hard to follow, is it _useful to draw the boundary_?

Rules that are simple and enforced automatically are easier to follow correctly than rules that are complex and unchecked.[^4]
However, if there's an asterisk on each memory access, then what's the point of the word "safety" anyway?

[^4]: It also doesn't mean that adhering to the rules is easy work - borrow checking can be a fickle goddess 🦀.

So, even though Fil-C _is safer_ than Yolo-C and the proposed Memory-Trapping-Zig _would be safer_ than Yolo-Zig,
they are not really safe in the only way that matters **to you**.

They still have unmarked and unchecked memory footguns that **you** need to check and be constantly aware of,
and calling that "memory safe" somewhat belittles the title-case "Memory Safety" that people are raving about (and expecting!).

## Conclusion

Memory Safety is great! It works **for you**: it makes life harder for someone else - the compiler author or the standard library designer - so that **you** can have a better time working on the things you care about! 

There can be some subtlety to the definition, and of course there are also issues unrelated to memory safety that are worth systematically attacking, but the point of having memory safety is to have _less_ subtlety overall.

Or, to borrow [Ralf Jung’s phrasing][what-the-hardware-does] slightly out of context: <mark>"When writing safe Rust, you do not have to worry"</mark> about the abstract machine. And really, the world provides enough things to worry about without adding that to the list.

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

Similarly, any language where FFI or unsafe memory access is an explicit construct (`unsafe extern "C"` in Rust, P/Invoke and `unsafe` in .Net, or Java's JNI),
_can_ be safe.

### Rust `/proc/self/mem`

A language is _not_ unsafe just because it lets you write to memory like this:

```rust
let mut f = fs::OpenOptions::new()
    .write(true)
    .open("/proc/self/mem").unwrap();
```

Yes, you can do really bad memory crimes with this. 
However, there's a clear sign: `/proc/self/mem` is a great way to say "all bets are off". 

### Zig is Unsafe

This is not controversial, but see [a previous post]({{< ref "/posts/2025-05-zig-mem-safety.md" >}}).
