# Exercise 03: Manual Edge Detection

Build a `Conv2d(1, 1, kernel_size=3, bias=False, padding=1)`. Manually set its
weight to the Sobel-X kernel:

```
[[-1, 0, 1],
 [-2, 0, 2],
 [-1, 0, 1]]
```

Create a small `(1, 1, 5, 5)` image with a vertical edge (left half = 0, right
half = 1). Run the conv and print the response — high magnitude at the edge.
