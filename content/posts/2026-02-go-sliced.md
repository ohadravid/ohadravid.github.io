---
title: "Sliced by Go's Slices"
summary: ""
date: 2026-02-26T16:00:00+00:00
tags: ["go", "python", "thoughts"]
type: post
showTableOfContents: false
image: "/2026-02-go-sliced/gopherNoMouthWide.webp"
weight: 1
---
<style>
.post .post-content {
    margin-top: 0px;
}
</style>
Today, I was sliced by Go's slices. Actually, by Go's variadics. Question: what does this snippet print?

```go
func main() {
	nums := []int{1, 2, 3}
	PrintSquares(nums...) // variadic expansion
	fmt.Printf("2 %v\n", nums)
}

func PrintSquares(nums ...int) {
	for i, n := range nums {
		nums[i] = n * n
	}
	fmt.Printf("1 %v\n", nums)
}
```

Answer ([Playground]):

```
1 [1 4 9]
2 [1 4 9]
```

🫠

[Playground]: https://go.dev/play/p/jy7gqxJUDNC

Meaning, in Go, when you use a slice for variadic expansion (`s...`),
and you use a variadic parameter to capture said slice (`paramSlice ...int`),
they are the same[^1] slice, and mutating one will mutate the other.

[^1]: They are _almost_ the same: they share the same _underlying array_. You do get a copy of the little `(ptr, len, capacity)` [struct][slice-struct] which is what a slice *is*. \
If you reassign the variable, e.g `nums = append(nums, 16)`, [that's a different ~story~ can of worms entirely.][why-append-modify]

[slice-struct]: https://go.dev/blog/slices-intro
[why-append-modify]: https://stackoverflow.com/questions/35920534/why-does-append-modify-passed-slice

In Python, you actually _can't_ do that because `*args` is always a tuple:

```python
def check(*args):
    args[1] = "hi zev" # TypeError: 'tuple' object does not support item assignment

l = [1, 2, 3]
check(*l)
print(l)
```

So the assignment fails, but even with `**kwargs`:

```python
def check(**kwargs):
    kwargs["1"] = "hi zev"

d = {"1": None}
check(**d)
assert d["1"] is None, "Sanity prevails! 😌"
```

We get a new dictionary in the callee, and sanity prevails.

And, TBH, I've been putting up with Go's... peculiarities for a while now, 
but every now and then there's something like this, where I feel like Go wants me to die an early death from high blood pressure.

Which it _probably_ doesn't. But I can't shake that feeling.

```bash
$ /usr/bin/time go build
       49.82 real        59.75 user        27.26 sys
$ 
```

Why don't you print anything, Go? WHY?

<p style="text-align:center;">
    <img src="/2026-02-go-sliced/gopherNoMouth.webp" 
        alt="The Go Gopher, sans Mouth" style="width:33%; height:auto;" width="1634" height="2224" />
</p>

why? 😔


--

_Discuss on [lobste.rs](https://lobste.rs/s/o3cpxf/sliced_by_go_s_slices)._

_If you liked this, you might also like [The story of the craziest `__init__` I’ve ever seen]({{< ref "/posts/2025-04-19-frank.md" >}})._

<!-- This is a good time for you to upvote and comment about the whimsical nature of Go's personality -->
