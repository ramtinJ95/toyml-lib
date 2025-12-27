package main

import "fmt"

type Value struct {
	Data float64
	Prev []*Value
}

func NewValue(data float64) *Value {
	return &Value{
		Data: data,
	}
}

func (v *Value) Stringify() {
	fmt.Printf("Value:(data = %f)\n", v.Data)
	fmt.Printf("Value:(data = %v)\n", v.Prev)
}

func (v *Value) Add(other *Value) *Value {
	return &Value{
		Data: v.Data + other.Data,
		Prev: []*Value{v, other},
	}
}

func (v *Value) Mul(other *Value) *Value {
	return &Value{
		Data: v.Data * other.Data,
		Prev: []*Value{v, other},
	}
}
