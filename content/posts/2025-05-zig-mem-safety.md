---
title: "Zig is not F'ing memory safe"
summary: ""
date: 2025-05-10T08:00:00+00:00
tags: ["zig"]
type: post
showTableOfContents: false
weight: 5
---
<style>
.post .post-content {
    margin-top: 0px;
}
</style>
It's not. I'm not saying that's _inherently bad_.
It's just what it is.
Consider the following code:

```zig
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

var list = try std.ArrayList(u8).initCapacity(allocator, 4);
try list.appendSlice("Hell");

const c = &list.items[0];

try list.append('o');
try list.append(c.*);
```

It compiles just fine but triggers a segfault due to a Use-After-Free bug in the last line:

```bash
$ zig run main.zig
Segmentation fault at address 0x104b00000
aborting due to recursive panic
zsh: abort      zig run main.zig
```


If you have a [use case](https://matklad.github.io/2023/03/26/zig-and-rust.html#TigerBeetle) where you don't care about that - 
you do you!

We can analyze what [safety even is](https://steveklabnik.com/writing/does-unsafe-undermine-rusts-guarantees/) (or what [memory even is](https://tratt.net/laurie/blog/2022/making_rust_a_better_fit_for_cheri_and_other_platforms.html)),
or say that other tools can help (ASan, etc.)
but this code isn't safer than similar code written in C, C++, and other unsafe languages.



