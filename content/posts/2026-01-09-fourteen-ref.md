---
title: "&&&&&&&&&&&&&&str"
summary: ""
date: 2026-01-09T14:00:00+00:00
tags: ["rust", "performance"]
type: post
showTableOfContents: false
weight: 1
---

While adding a failing test to the Rust compiler, I stumbled upon a peculiar [code generation test](https://github.com/rust-lang/rust/blob/1.92.0/tests/codegen-llvm/issues/str-to-string-128690.rs):

```rust
pub fn thirteen_ref(input: &&&&&&&&&&&&&str) -> String {
    // CHECK-NOT: {{(call|invoke)}}{{.*}}@{{.*}}core{{.*}}fmt{{.*}}
    input.to_string()
}

// This is a known performance cliff because of the macro-generated
// specialized impl. If this test suddenly starts failing,
// consider removing the `to_string_str!` macro in `alloc/str/string.rs`.
//
pub fn fourteen_ref(input: &&&&&&&&&&&&&&str) -> String {
    // CHECK: {{(call|invoke)}}{{.*}}@{{.*}}core{{.*}}fmt{{.*}}
    input.to_string()
}
```

In case you are wondering where are the tests here: the `CHECK` and `CHECK-NOT` comments are actually the test assertions, which are tested using [LLVM's FileCheck framework](https://llvm.org/docs/CommandGuide/FileCheck.html).

Opening up [godbolt](https://godbolt.org/z/ex5WexToz), the former allocates a new string and calls `memcpy`,
while the latter calls, as the the test suggests, to `<str as core::fmt::Display>::fmt`, which is less efficient. But why 14? 

The `to_string_str` macro has moved since that comment was written, and now lives at [library/alloc/src/string.rs](https://github.com/rust-lang/rust/blob/1.92.0/library/alloc/src/string.rs#L3013):

```rust
macro_rules! to_string_str {
    {$($type:ty,)*} => {
        $(
            impl SpecToString for $type {
                #[inline]
                fn spec_to_string(&self) -> String {
                    let s: &str = self;
                    String::from(s)
                }
            }
        )*
    };
}

to_string_str! {
    Cow<'_, str>,
    String,
    &&&&&&&&&&&&str,
    &&&&&&&&&&&str,
    &&&&&&&&&&str,
    &&&&&&&&&str,
    &&&&&&&&str,
    &&&&&&&str,
    &&&&&&str,
    &&&&&str,
    &&&&str,
    &&&str,
    &&str,
    &str,
    str,
}
```

No special generics, no secret compiler internals voodoo, no crazy type system magic - just plain old copy-paste until we get what we want. I _love_ it.

So why 14? \
Seems like because Rust thinks that 13 references [ought to be enough for anybody](https://www.computerworld.com/article/1563853/the-640k-quote-won-t-go-away-but-did-gates-really-say-it.html).

_Edit: More interesting details (for example, did you notice there are only **12 &s** in `to_string_str!`?) in the [lobste.rs discussion](https://lobste.rs/s/jo3dnh/str), and more joking over at [r/rust](https://www.reddit.com/r/rust/comments/1q8czqi/str/)._