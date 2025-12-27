package main

import "fmt"

func main() {
	fmt.Println("hello world")
	a := NewValue(3)
	b := NewValue(2)
	c := a.Mul(b)
	c.Stringify()
	d := c.Add(a)
	d.Stringify()
}
